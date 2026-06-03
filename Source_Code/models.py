# ============================================================
# MODELS
# ============================================================

import torch.nn as nn

from torchvision import models


# ============================================================
# DENSENET121
# ============================================================

def build_densenet121(num_classes=3):

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ============================================================
# EFFICIENTNET B0
# ============================================================

def build_efficientnet_b0(num_classes=3):

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(
    model_name,
    num_classes=3
):

    if model_name == "DenseNet121":

        return build_densenet121(
            num_classes=num_classes
        )

    elif model_name == "EfficientNetB0":

        return build_efficientnet_b0(
            num_classes=num_classes
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )