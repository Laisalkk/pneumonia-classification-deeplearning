# -*- coding: utf-8 -*-

# ============================================================
# AUGMENTATION MODULE
# ============================================================

from torchvision import transforms
from config import IMG_SIZE


# ============================================================
# TRAIN TRANSFORM
# ============================================================

def get_train_transform():
    """
    Transformasi untuk data training.

    Termasuk:
    - Convert grayscale -> 3 channel
    - Resize
    - Data augmentation ringan
    - Normalisasi ImageNet
    """

    train_transform = transforms.Compose([

        # ----------------------------------------------------
        # Convert 1 channel menjadi 3 channel
        # ----------------------------------------------------
        transforms.Grayscale(
            num_output_channels=3
        ),

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        # ----------------------------------------------------
        # Rotasi ringan
        # ----------------------------------------------------
        transforms.RandomRotation(
            degrees=10
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        # ----------------------------------------------------
        # Translasi & zoom ringan
        # ----------------------------------------------------
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------
        transforms.ToTensor(),

        # ----------------------------------------------------
        # Normalisasi ImageNet
        # Digunakan karena memakai pretrained model
        # ----------------------------------------------------
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return train_transform


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

def get_validation_transform():
    """
    Transformasi validation.

    Tidak menggunakan augmentasi.
    """

    validation_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return validation_transform


# ============================================================
# TEST TRANSFORM
# ============================================================

def get_test_transform():
    """
    Transformasi test.

    Sama dengan validation.
    """

    test_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return test_transform


# ============================================================
# GET ALL TRANSFORMS
# ============================================================

def get_transforms():
    """
    Mengembalikan seluruh transform sekaligus.
    """

    return (
        get_train_transform(),
        get_validation_transform(),
        get_test_transform()
    )