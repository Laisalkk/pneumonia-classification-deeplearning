import torch
import torch.nn as nn
from torchvision import models

def get_inference_model(model_name, num_classes=3):
    """
    Membangun arsitektur model tanpa mendownload pretrained weights asli ImageNet,
    karena kita akan memasukkan bobot custom hasil training kita sendiri.
    """
    if model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} tidak dikenali.")
        
    return model
