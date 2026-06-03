
# ============================================================
# EVALUATION MODULE
# ============================================================

import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import (
    label_binarize
)

from config import *

# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    dataloader,
    device
):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    progress_bar = tqdm(
        dataloader,
        desc="Testing",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            _, preds = torch.max(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

            y_prob.extend(
                probabilities.cpu().numpy()
            )

    return (

        np.array(y_true),

        np.array(y_pred),

        np.array(y_prob)

    )

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(

    y_true,
    y_pred,
    model_name

):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_percent = (

        cm.astype(float)

        / cm.sum(axis=1)[:, np.newaxis]

    ) * 100

    annotations = np.empty_like(
        cm
    ).astype(str)

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            annotations[i, j] = (

                f"{cm[i,j]}\n"

                f"{cm_percent[i,j]:.1f}%"

            )

    plt.figure(figsize=(8, 6))

    sns.heatmap(

        cm,

        annot=annotations,

        fmt="",

        cmap="Blues",

        xticklabels=list(
            CLASS_NAMES.values()
        ),

        yticklabels=list(
            CLASS_NAMES.values()
        )

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    save_path = os.path.join(

        RESULT_DIR,

        f"confusion_matrix_{model_name}.png"

    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] {save_path}"
    )

# ============================================================
# SAVE ROC CURVE
# ============================================================

def save_multiclass_roc(

    y_true,
    y_prob,
    model_name

):

    classes = np.arange(
        NUM_CLASSES
    )

    y_true_bin = label_binarize(

        y_true,

        classes=classes

    )

    plt.figure(figsize=(8, 6))

    roc_auc_scores = {}

    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(

            y_true_bin[:, i],

            y_prob[:, i]

        )

        roc_auc = auc(
            fpr,
            tpr
        )

        roc_auc_scores[i] = roc_auc

        plt.plot(

            fpr,

            tpr,

            linewidth=2,

            label=(
                f"{CLASS_NAMES[i]}"
                f" (AUC={roc_auc:.3f})"
            )

        )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend()

    save_path = os.path.join(

        RESULT_DIR,

        f"roc_curve_{model_name}.png"

    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] {save_path}"
    )

    return roc_auc_scores

# ============================================================
# SAVE CLASSIFICATION REPORT CSV
# ============================================================

def save_classification_report_csv(

    y_true,
    y_pred,
    model_name

):

    report_dict = classification_report(

        y_true,

        y_pred,

        target_names=list(
            CLASS_NAMES.values()
        ),

        output_dict=True

    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    save_path = os.path.join(

        RESULT_DIR,

        f"classification_report_{model_name}.csv"

    )

    report_df.to_csv(
        save_path
    )

    print(
        f"[SAVED] {save_path}"
    )

# ============================================================
# SAVE METRICS CSV
# ============================================================

def save_metrics_csv(

    metrics_dict,
    model_name

):

    save_path = os.path.join(

        RESULT_DIR,

        f"metrics_{model_name}.csv"

    )

    pd.DataFrame(
        [metrics_dict]
    ).to_csv(
        save_path,
        index=False
    )

    print(
        f"[SAVED] {save_path}"
    )

# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(

    model,
    test_loader,
    device,
    model_name

):

    print("\n")
    print("=" * 60)

    print(
        f"EVALUATION : {model_name}"
    )

    print("=" * 60)

    y_true, y_pred, y_prob = get_predictions(

        model=model,

        dataloader=test_loader,

        device=device

    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(

        y_true,

        y_pred,

        average="macro"

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="macro"

    )

    macro_f1 = f1_score(

        y_true,

        y_pred,

        average="macro"

    )

    weighted_f1 = f1_score(

        y_true,

        y_pred,

        average="weighted"

    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"Macro F1  : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print("\nClassification Report\n")

    report = classification_report(

        y_true,

        y_pred,

        target_names=list(
            CLASS_NAMES.values()
        )

    )

    print(report)

    save_classification_report_csv(

        y_true,

        y_pred,

        model_name

    )

    save_confusion_matrix(

        y_true,

        y_pred,

        model_name

    )

    roc_auc_scores = save_multiclass_roc(

        y_true,

        y_prob,

        model_name

    )

    metrics = {

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "Macro_F1":
            macro_f1,

        "Weighted_F1":
            weighted_f1,

        "AUC_No_Disease":
            roc_auc_scores[0],

        "AUC_Bacterial":
            roc_auc_scores[1],

        "AUC_Viral":
            roc_auc_scores[2]

    }

    save_metrics_csv(
        metrics,
        model_name
    )

    return metrics

