# ============================================================
# MAIN PIPELINE
# ============================================================

import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# ============================================================
# PROJECT DIRECTORY
# ============================================================
PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/MDS"

# ============================================================
# PROJECT MODULES
# ============================================================
from config import *
from preprocessing import (
    load_metadata,
    create_image_paths,
    validate_images,
    dataset_summary,
    split_dataset,
    split_summary,
    get_class_weights
)
from augmentation import get_transforms
from dataset import PneumoniaDataset
from models import get_model
from train_model import train_model
from evaluate_model import evaluate_model

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("PNEUMONIA CLASSIFICATION PIPELINE - AUTOMATED GRID SEARCH")
    print("=" * 60)

    # Set seed awal
    set_seed(SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device : {device}")

    # Paths Setup
    DATA_DIR = os.path.join(PROJECT_DIR, "Dataset")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
    EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "Experiments")

    # --------------------------------------------------------
    # AUTOMATIC DIRECTORY CREATION (SAFETY NET)
    # --------------------------------------------------------
    print("\n[INFO] Checking workplace directories...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    print("--> All directories checked and ready.")

    TRAIN_DIR = os.path.join(DATA_DIR, TRAIN_FOLDER)
    CSV_PATH = os.path.join(DATA_DIR, LABEL_FILE)

    # Pipeline Preprocessing Awal (Satu kali jalan untuk seluruh eksperimen)
    print("\n[STEP 1] Loading and Validating Metadata...")
    df = load_metadata(CSV_PATH)
    df = create_image_paths(df, TRAIN_DIR)
    df = validate_images(df)
    dataset_summary(df)

    print("\n[STEP 2] Splitting Dataset...")
    train_df, val_df, test_df = split_dataset(
        df=df, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=SEED
    )
    split_summary(train_df, val_df, test_df)

    class_weights = get_class_weights(train_df, device)
    train_transform, val_transform, test_transform = get_transforms()

    train_dataset = PneumoniaDataset(dataframe=train_df, transform=train_transform)
    val_dataset = PneumoniaDataset(dataframe=val_df, transform=val_transform)
    test_dataset = PneumoniaDataset(dataframe=test_df, transform=test_transform)

    all_results = []
    
    # --------------------------------------------------------
    # CONTINGENCY: RESUME/SKIP COMPLETED RUNS (SAFETY NET)
    # --------------------------------------------------------
    final_csv_path = os.path.join(RESULTS_DIR, "final_results.csv")
    completed_runs = set()

    if os.path.exists(final_csv_path):
        try:
            old_df = pd.read_csv(final_csv_path)
            # Ambil kombinasi Experiment_ID dan Model yang sudah sukses ditraining sebelumnya
            for _, row in old_df.iterrows():
                completed_runs.add((row["Experiment_ID"], row["Model"]))
            all_results = old_df.to_dict(orient="records")
            print(f"\n[SAFETY NET] Found existing final_results.csv. Loaded {len(completed_runs)} completed runs.")
        except Exception as e:
            print(f"[WARNING] Failed to load existing results spreadsheet, starting fresh: {e}")

    # Menentukan target list eksperimen berdasarkan mode DEBUG
    active_experiments = EXPERIMENTS if not DEBUG else [
        {"id": "DEBUG_E1", "lr": 0.001, "batch_size": 8, "epochs": 1}
    ]
    
    total_runs = len(active_experiments) * len(MODEL_NAMES)
    current_run = 1

    # ========================================================
    # AUTOMATED EXPERIMENT LOOP (GRID SEARCH)
    # ========================================================
    for exp in active_experiments:
        exp_id = exp["id"]
        exp_lr = exp["lr"]
        exp_batch = exp["batch_size"]
        exp_epochs = exp["epochs"]

        print("\n" + "#" * 70)
        print(f" CONFIGURING EXPERIMENT: {exp_id} | LR: {exp_lr} | Batch Size: {exp_batch} | Epochs: {exp_epochs}")
        print("#" * 70)

        # RE-CREATE DATALOADER DENGAN BATCH SIZE DINAMIS SESUAI EKSPERIMEN
        train_loader = DataLoader(
            train_dataset, batch_size=exp_batch, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset, batch_size=exp_batch, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        test_loader = DataLoader(
            test_dataset, batch_size=exp_batch, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )

        for model_name in MODEL_NAMES:
            unique_model_run_name = f"{exp_id}_{model_name}"

            # --- CHECK JIKA MODEL INI SUDAH PERNAH TRAINING SEBELUMNYA (SAFETY CHECK) ---
            if (exp_id, model_name) in completed_runs:
                print(f"[{current_run}/{total_runs}] SKIPPING: {unique_model_run_name} (Already completed in previous session).")
                current_run += 1
                continue
            # ----------------------------------------------------------------------------

            # Tetapkan seed kembali agar inisialisasi bobot awal konisten antar eksperimen model yang sama
            set_seed(SEED)
            
            print("\n" + "=" * 60)
            print(f"[{current_run}/{total_runs}] RUNNING: {unique_model_run_name}")
            print("=" * 60)

            # Build Model
            model = get_model(model_name=model_name, num_classes=NUM_CLASSES)
            model = model.to(device)

            # Training Menggunakan Parameter Eksperimen yang Dilempar Langsung
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                class_weights=class_weights,
                model_name=unique_model_run_name,
                learning_rate=exp_lr,
                num_epochs=exp_epochs,
                experiment_id=exp_id
            )

            # Evaluation
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                model_name=unique_model_run_name
            )

            # Tambahkan metadata eksperimen ke penampung hasil akhir untuk final_results.csv
            metrics["Experiment_ID"] = exp_id
            metrics["LR"] = exp_lr
            metrics["Batch_Size"] = exp_batch
            all_results.append(metrics)
            
            # --- UPDATE SPREADSHEET REALTIME TIAP SELESAI 1 MODEL ---
            results_df = pd.DataFrame(all_results)
            cols = ["Experiment_ID", "Model"] + [c for c in results_df.columns if c not in ["Experiment_ID", "Model"]]
            results_df = results_df[cols]
            results_df.to_csv(final_csv_path, index=False)
            
            current_run += 1

    print(f"\n[SAVED] Final comparison spreadsheet updated at: {final_csv_path}")
    print("\n" + "=" * 60)
    print("ALL TARGET EXPERIMENTAL COMBINATIONS SUCCESSFULLY EXECUTED")
    print("=" * 60)

if __name__ == "__main__":
    main()