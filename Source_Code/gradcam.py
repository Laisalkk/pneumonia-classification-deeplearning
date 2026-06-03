# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import *
from preprocessing import load_metadata, create_image_paths, validate_images, split_dataset
from augmentation import get_transforms
from dataset import PneumoniaDataset
from models import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class):
        output = self.model(input_image)
        self.model.zero_grad()
        
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap

def run_gradcam_analysis():
    print("=" * 60)
    print("RUNNING EXPLAINABLE AI (XAI) - GRAD-CAM ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/MDS"
    CSV_PATH = os.path.join(BASE_DIR, "Dataset", LABEL_FILE)
    TRAIN_DIR = os.path.join(BASE_DIR, "Dataset", TRAIN_FOLDER)
    RESULTS_DIR = os.path.join(BASE_DIR, "Results")
    XAI_DIR = os.path.join(RESULTS_DIR, "XAI_GRADCAM")
    os.makedirs(XAI_DIR, exist_ok=True)

    df = load_metadata(CSV_PATH)
    df = create_image_paths(df, TRAIN_DIR)
    df = validate_images(df)
    _, _, test_df = split_dataset(df, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED)

    _, _, test_transform = get_transforms()
    test_dataset = PneumoniaDataset(dataframe=test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # PERBAIKAN NAMA FILE: Ditambahkan _best.pth sesuai Drive Anda
    checkpoint_path = os.path.join(RESULTS_DIR, "checkpoints", "E5_EfficientNetB0_best.pth")
    model = get_model(model_name="EfficientNetB0", num_classes=NUM_CLASSES)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] Loaded model weights from {checkpoint_path}")
    else:
        print(f"[ERROR] Checkpoint tidak ditemukan!")
        return

    model = model.to(device)

    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    correct_samples = []
    wrong_samples = []

    for idx, (image, label) in enumerate(test_loader):
        img_device = image.to(device)
        output = model(img_device)
        pred = torch.argmax(output, dim=1).item()
        true_label = label.item()

        if pred == true_label and len(correct_samples) < 5:
            correct_samples.append((idx, image, true_label, pred))
        elif pred != true_label and len(wrong_samples) < 5:
            wrong_samples.append((idx, image, true_label, pred))

        if len(correct_samples) == 5 and len(wrong_samples) == 5:
            break

    categories = [("CORRECT", correct_samples), ("WRONG", wrong_samples)]
    for cat_name, samples in categories:
        for rank, (origin_idx, img_tensor, true_lbl, pred_lbl) in enumerate(samples):
            img_device = img_tensor.to(device)
            heatmap = cam.generate_heatmap(img_device, pred_lbl)

            img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = np.uint8(255 * img_np)

            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(overlayed_img)
            axes[1].set_title("Grad-CAM Activation Map")
            axes[1].axis('off')

            true_name = CLASS_NAMES[true_lbl]
            pred_name = CLASS_NAMES[pred_lbl]
            plt.suptitle(f"Grad-CAM ({cat_name}) | True: {true_name} | Pred: {pred_name}", fontsize=12)
            
            save_path = os.path.join(XAI_DIR, f"{cat_name.lower()}_gradcam_sample_{rank+1}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"--> Saved Grad-CAM Plot: {save_path}")

if __name__ == '__main__':
    run_gradcam_analysis()
