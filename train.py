from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    WeightedRandomSampler = None

from feature_engineering import (
    build_tabular_features,
    fixed_background_predictions as fixed_tabular_background_predictions,
)
try:
    from load_data import EEGDataset
    from model import Net
except ModuleNotFoundError:
    EEGDataset = None
    Net = None

from utils import get_device, set_seed

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RES_DIR = PROJECT_ROOT / "res"

CONFIG = {
    "train_dir": DATA_ROOT / "train",
    "train_label_csv": DATA_ROOT / "train_labels.csv",
    "batch_size": 64,
    "max_epochs": 70,
    "learning_rate": 8e-4,
    "weight_decay": 8e-3,
    "seed": 42,
    "num_workers": 0,
    "use_cpu": False,
    "background_per_subject_session": 90,
    "prediction_background_count": 100,
    "selected_tabular_model": "ensemble_mean",
    "decision_threshold": 0.445,
    "prediction_rule": "threshold",
    "train_neural_model": False,
}


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["eeg_file"].str.extract(r"sub(\d+)_sess(\d+)_epoch(\d+)")
    parsed.columns = ["sub", "sess", "epoch"]
    parsed = parsed.astype(int)
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


def augment_batch(eeg: torch.Tensor) -> torch.Tensor:
    if eeg.size(0) == 0:
        return eeg
    noise = 0.02 * torch.randn_like(eeg)
    apply_noise = (torch.rand(eeg.size(0), 1, 1, 1, device=eeg.device) < 0.70).float()
    eeg = eeg + apply_noise * noise * eeg.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    if torch.rand(1, device=eeg.device).item() < 0.50:
        shift = int(torch.randint(-6, 7, (1,), device=eeg.device).item())
        eeg = torch.roll(eeg, shifts=shift, dims=-1)
    return eeg


def build_session_split(label_csv: Path) -> Tuple[List[int], List[int]]:
    df = add_metadata(pd.read_csv(label_csv))
    train_indices = df.index[df["sess"] == 1].tolist()
    val_indices = df.index[df["sess"] == 2].tolist()
    return train_indices, val_indices


def make_loader(
    dataset: EEGDataset,
    labels: Iterable[int] | None,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def build_sampler(label_names: pd.Series, selected_indices: List[int]) -> WeightedRandomSampler:
    selected_labels = label_names.iloc[selected_indices].map({"background": 0, "target": 1}).to_numpy()
    counts = np.bincount(selected_labels, minlength=2)
    weights = 1.0 / counts[selected_labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_all: List[int] = []
    probs_all: List[float] = []

    with torch.no_grad():
        for eeg, labels in dataloader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(eeg)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    labels_np = np.asarray(labels_all)
    probs_np = np.asarray(probs_all)
    preds_np = (probs_np >= 0.5).astype(int)
    accuracy = float((preds_np == labels_np).mean())
    return total_loss / len(labels_np), accuracy, probs_np, labels_np


def fixed_background_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    background_count: int,
) -> np.ndarray:
    pred = np.ones(len(df), dtype=int)
    work = df.copy()
    work["prob_target"] = probs
    for _, group in work.groupby(["sub", "sess"], sort=False):
        sorted_index = group.sort_values("prob_target").index.to_numpy()
        n_background = min(background_count, len(sorted_index))
        local_positions = work.index.get_indexer(sorted_index[:n_background])
        pred[local_positions] = 0
    return pred


def select_stable_threshold(
    split_predictions: List[Tuple[str, np.ndarray, np.ndarray]],
    thresholds: np.ndarray | None = None,
) -> Tuple[float, List[Dict[str, float | str]]]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.35, 0.6001, 0.005), 3)

    rows: List[Dict[str, float | str]] = []
    best_key: Tuple[float, float, float] | None = None
    best_threshold = float(CONFIG["decision_threshold"])
    for threshold in thresholds:
        split_scores = {
            split_name: accuracy_score(labels, probs >= threshold)
            for split_name, labels, probs in split_predictions
        }
        min_accuracy = min(split_scores.values())
        mean_accuracy = float(np.mean(list(split_scores.values())))
        # Prefer the threshold that raises the weaker split.  On ties, prefer
        # higher mean accuracy and then the less aggressive shift from 0.50.
        key = (min_accuracy, mean_accuracy, -abs(float(threshold) - 0.50))
        rows.append(
            {
                "threshold": float(threshold),
                "min_accuracy": min_accuracy,
                "mean_accuracy": mean_accuracy,
                **split_scores,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def train_one_run(
    train_indices: List[int],
    val_indices: List[int] | None,
    epochs: int,
    seed: int,
    save_history: bool,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]], int, float]:
    if torch is None or EEGDataset is None or Net is None:
        raise RuntimeError("PyTorch is required when CONFIG['train_neural_model'] is True.")

    set_seed(seed)
    device = get_device(CONFIG["use_cpu"])
    label_df = add_metadata(pd.read_csv(CONFIG["train_label_csv"]))
    label_series = label_df["label"]

    train_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        normalize=False,
        selected_indices=train_indices,
    )
    train_loader = make_loader(
        train_dataset,
        labels=None,
        batch_size=CONFIG["batch_size"],
        sampler=build_sampler(label_series, train_indices),
    )

    val_loader = None
    val_df = None
    if val_indices is not None:
        val_dataset = EEGDataset(
            data_dir=CONFIG["train_dir"],
            label_csv=CONFIG["train_label_csv"],
            normalize=False,
            selected_indices=val_indices,
        )
        val_loader = make_loader(val_dataset, labels=None, batch_size=256, shuffle=False)
        val_df = label_df.iloc[val_indices].reset_index(drop=True)

    sample_eeg, _ = train_dataset[0]
    model = Net(input_shape=tuple(sample_eeg.shape)).to(device)
    class_counts = label_series.iloc[train_indices].value_counts()
    class_weights = torch.tensor(
        [
            len(train_indices) / (2.0 * class_counts["background"]),
            len(train_indices) / (2.0 * class_counts["target"]),
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_epoch = epochs
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for eeg, labels in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            eeg = augment_batch(eeg)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(eeg)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        scheduler.step()
        train_loss = running_loss / seen
        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None and val_df is not None:
            val_loss, val_acc, probs, labels_np = evaluate(model, val_loader, criterion, device)
            calibrated = fixed_background_predictions(
                val_df,
                probs,
                CONFIG["background_per_subject_session"],
            )
            calibrated_acc = float((calibrated == labels_np).mean())
            row.update(
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "calibrated_val_acc": calibrated_acc,
                }
            )
            score_for_selection = max(val_acc, calibrated_acc)
            if score_for_selection > best_val_acc:
                best_val_acc = score_for_selection
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_shape": tuple(sample_eeg.shape),
                        "background_per_subject_session": CONFIG["background_per_subject_session"],
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                    },
                    MODEL_DIR / "best_model.pth",
                )
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | calibrated_val_acc={calibrated_acc:.4f} | best saved"
                )
            elif epoch == 1 or epoch % 5 == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | calibrated_val_acc={calibrated_acc:.4f}"
                )
        elif epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        history.append(row)

    if val_loader is None:
        best_state = deepcopy(model.state_dict())
        best_epoch = epochs
        best_val_acc = float("nan")

    if save_history:
        RES_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(RES_DIR / "training_history.csv", index=False)

    return best_state, history, best_epoch, best_val_acc


def make_tabular_models() -> Dict[str, object]:
    models = {}
    base_config = dict(
        num_leaves=10,
        learning_rate=0.025,
        subsample=0.92,
        colsample_bytree=0.82,
        reg_lambda=2.0,
        reg_alpha=0.15,
        min_child_samples=10,
        verbose=-1,
        n_jobs=-1,
    )
    for seed in range(5):
        models[f"lgbm_mid_{seed}"] = LGBMClassifier(n_estimators=300, random_state=seed, **base_config)
    return models


def train_tabular_ensemble(label_df: pd.DataFrame) -> None:
    set_seed(CONFIG["seed"])
    meta_df = add_metadata(label_df)
    features = build_tabular_features(meta_df, CONFIG["train_dir"])
    labels = (meta_df["label"] == "target").astype(int).to_numpy()

    validation_rows: List[Dict[str, float | str]] = []
    ensemble_split_predictions: List[Tuple[str, np.ndarray, np.ndarray]] = []
    split_specs = {
        "session1_to_session2": (
            meta_df.index[meta_df["sess"] == 1].to_numpy(),
            meta_df.index[meta_df["sess"] == 2].to_numpy(),
        ),
        "session2_to_session1": (
            meta_df.index[meta_df["sess"] == 2].to_numpy(),
            meta_df.index[meta_df["sess"] == 1].to_numpy(),
        ),
    }

    for split_name, (train_idx, val_idx) in split_specs.items():
        split_probs: List[np.ndarray] = []
        for model_name, model in make_tabular_models().items():
            model.fit(features[train_idx], labels[train_idx])
            prob = model.predict_proba(features[val_idx])[:, 1]
            split_probs.append(prob)
            pred = (prob >= 0.5).astype(int)
            calibrated = fixed_tabular_background_predictions(
                meta_df.iloc[val_idx].reset_index(drop=True),
                prob,
                CONFIG["background_per_subject_session"],
            )
            rank_calibrated = fixed_tabular_background_predictions(
                meta_df.iloc[val_idx].reset_index(drop=True),
                prob,
                CONFIG["prediction_background_count"],
            )
            threshold_pred = (prob >= CONFIG["decision_threshold"]).astype(int)
            row = {
                "split": split_name,
                "model": model_name,
                "accuracy": accuracy_score(labels[val_idx], pred),
                "threshold_accuracy": accuracy_score(labels[val_idx], threshold_pred),
                "auc": roc_auc_score(labels[val_idx], prob),
                "calibrated_accuracy": accuracy_score(labels[val_idx], calibrated),
                "rank_calibrated_accuracy": accuracy_score(labels[val_idx], rank_calibrated),
            }
            validation_rows.append(row)
            print(
                f"{split_name} | {model_name} | "
                f"acc={row['accuracy']:.4f} | auc={row['auc']:.4f} | "
                f"calibrated_acc={row['calibrated_accuracy']:.4f}"
            )

        ensemble_prob = np.mean(split_probs, axis=0)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        ensemble_threshold_pred = (ensemble_prob >= CONFIG["decision_threshold"]).astype(int)
        ensemble_calibrated = fixed_tabular_background_predictions(
            meta_df.iloc[val_idx].reset_index(drop=True),
            ensemble_prob,
            CONFIG["background_per_subject_session"],
        )
        row = {
            "split": split_name,
            "model": "ensemble_mean",
            "accuracy": accuracy_score(labels[val_idx], ensemble_pred),
            "threshold_accuracy": accuracy_score(labels[val_idx], ensemble_threshold_pred),
            "auc": roc_auc_score(labels[val_idx], ensemble_prob),
            "calibrated_accuracy": accuracy_score(labels[val_idx], ensemble_calibrated),
        }
        validation_rows.append(row)
        ensemble_split_predictions.append((split_name, labels[val_idx], ensemble_prob))
        print(
            f"{split_name} | ensemble_mean | "
            f"acc={row['accuracy']:.4f} | threshold_acc={row['threshold_accuracy']:.4f} | "
            f"auc={row['auc']:.4f} | "
            f"calibrated_acc={row['calibrated_accuracy']:.4f}"
        )

    decision_threshold, threshold_rows = select_stable_threshold(ensemble_split_predictions)
    split_prediction_map = {
        split_name: (split_labels, split_probs)
        for split_name, split_labels, split_probs in ensemble_split_predictions
    }
    for row in validation_rows:
        if row["model"] == "ensemble_mean":
            split_labels, split_probs = split_prediction_map[str(row["split"])]
            row["threshold_accuracy"] = accuracy_score(split_labels, split_probs >= decision_threshold)
    pd.DataFrame(threshold_rows).to_csv(RES_DIR / "threshold_scan.csv", index=False)
    print(f"Selected decision threshold: {decision_threshold:.3f}")

    final_models = make_tabular_models()
    for model in final_models.values():
        model.fit(features, labels)

    joblib.dump(
        {
            "models": final_models,
            "background_per_subject_session": CONFIG["background_per_subject_session"],
            "prediction_background_count": CONFIG["prediction_background_count"],
            "selected_model": CONFIG["selected_tabular_model"],
            "decision_threshold": decision_threshold,
            "prediction_rule": CONFIG["prediction_rule"],
            "feature_count": features.shape[1],
            "validation": validation_rows,
            "threshold_scan": threshold_rows,
        },
        MODEL_DIR / "tabular_ensemble.joblib",
    )
    pd.DataFrame(validation_rows).to_csv(RES_DIR / "tabular_validation.csv", index=False)
    print(f"Tabular ensemble saved to: {MODEL_DIR / 'tabular_ensemble.joblib'}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(CONFIG["train_label_csv"])
    train_tabular_ensemble(label_df)

    if not CONFIG["train_neural_model"]:
        print("Tabular ensemble is the primary model. Neural training skipped by configuration.")
        return

    train_indices, val_indices = build_session_split(CONFIG["train_label_csv"])
    best_state, _, best_epoch, best_val_acc = train_one_run(
        train_indices=train_indices,
        val_indices=val_indices,
        epochs=CONFIG["max_epochs"],
        seed=CONFIG["seed"],
        save_history=True,
    )

    print(f"Validation finished. best_epoch={best_epoch}, best_score={best_val_acc:.4f}")
    final_epochs = max(12, min(best_epoch, 35))
    all_indices = list(range(len(label_df)))
    final_state, _, _, _ = train_one_run(
        train_indices=all_indices,
        val_indices=None,
        epochs=final_epochs,
        seed=CONFIG["seed"] + 100,
        save_history=False,
    )

    sample_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        normalize=False,
    )
    sample_eeg, _ = sample_dataset[0]
    torch.save(
        {
            "model_state_dict": final_state,
            "input_shape": tuple(sample_eeg.shape),
            "background_per_subject_session": CONFIG["background_per_subject_session"],
            "best_epoch": final_epochs,
            "validation_best_score": best_val_acc,
        },
        MODEL_DIR / "final_model.pth",
    )
    torch.save(
        {
            "model_state_dict": final_state,
            "input_shape": tuple(sample_eeg.shape),
            "background_per_subject_session": CONFIG["background_per_subject_session"],
            "best_epoch": final_epochs,
            "validation_best_score": best_val_acc,
        },
        MODEL_DIR / "best_model.pth",
    )
    print(f"Final model saved. final_epochs={final_epochs}")


if __name__ == "__main__":
    main()
