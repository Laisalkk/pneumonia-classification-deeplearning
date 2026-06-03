# -*- coding: utf-8 -*-

# ============================================================
# PREPROCESSING MODULE
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch

# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(csv_path):
    """
    Load metadata CSV dan standarisasi nama kolom.
    """

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "file_name": "filename",
            "class_id": "label"
        }
    )

    required_columns = [
        "filename",
        "label"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset."
            )

    return df


# ============================================================
# CREATE IMAGE PATH
# ============================================================

def create_image_paths(df, train_dir):
    """
    Membuat path lengkap ke setiap file gambar.
    """

    df = df.copy()

    df["image_path"] = df["filename"].apply(
        lambda x: os.path.join(train_dir, x)
    )

    return df


# ============================================================
# VALIDATE IMAGE FILES
# ============================================================

def validate_images(df):
    """
    Memastikan file gambar benar-benar ada.
    """

    df = df.copy()

    df["file_exists"] = df["image_path"].apply(
        os.path.exists
    )

    missing_count = (~df["file_exists"]).sum()

    print("=" * 50)
    print("IMAGE VALIDATION")
    print("=" * 50)
    print(f"Missing files : {missing_count}")

    valid_df = df[
        df["file_exists"]
    ].copy()

    valid_df.reset_index(
        drop=True,
        inplace=True
    )

    print(f"Valid files   : {len(valid_df)}")

    return valid_df


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df):
    """
    Menampilkan informasi dataset.
    """

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print("\nDataset Shape")
    print(df.shape)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nClass Distribution")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nClass Percentage (%)")
    print(
        round(
            df["label"]
            .value_counts(normalize=True)
            .sort_index() * 100,
            2
        )
    )


# ============================================================
# TRAIN VALIDATION TEST SPLIT
# ============================================================

def split_dataset(
    df,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Split dataset menjadi:
    Train = 70%
    Validation = 15%
    Test = 15%
    """

    if round(train_size + val_size + test_size, 2) != 1.00:
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# SPLIT SUMMARY
# ============================================================

def split_summary(
    train_df,
    val_df,
    test_df
):
    """
    Menampilkan distribusi data hasil split.
    """

    print("=" * 50)
    print("DATA SPLIT SUMMARY")
    print("=" * 50)

    print(f"\nTrain      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nTRAIN DISTRIBUTION")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nVALIDATION DISTRIBUTION")
    print(
        val_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nTEST DISTRIBUTION")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

def get_class_weights(
    train_df,
    device
):
    """
    Menghitung class weight untuk mengatasi imbalance.
    """

    classes = np.sort(
        train_df["label"].unique()
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"]
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)

    print("=" * 50)
    print("CLASS WEIGHTS")
    print("=" * 50)
    print(class_weights)

    return class_weights
    