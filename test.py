from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None

from feature_engineering import (
    INDEX_TO_LABEL,
    add_metadata,
    build_tabular_features,
    fixed_background_predictions,
)
try:
    from load_data import EEGDataset
    from model import Net
except ModuleNotFoundError:
    EEGDataset = None
    Net = None

from utils import get_device

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RES_DIR = PROJECT_ROOT / "res"

CONFIG = {
    "test_dir": DATA_ROOT / "test",
    "tabular_model_path": MODEL_DIR / "tabular_ensemble.joblib",
    "neural_model_path": MODEL_DIR / "best_model.pth",
    "output_csv": RES_DIR / "predictions.csv",
    "probability_csv": RES_DIR / "prediction_probabilities.csv",
    "batch_size": 128,
    "num_workers": 0,
    "use_cpu": False,
    "prediction_rule": "threshold",
    "decision_threshold": 0.445,
    "use_neural_fallback": False,
}


def predict_with_tabular_model(test_df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(CONFIG["tabular_model_path"])
    features = build_tabular_features(test_df, CONFIG["test_dir"])
    selected_model = bundle.get("selected_model", "lgbm_ranker")
    if selected_model in bundle["models"]:
        target_prob = bundle["models"][selected_model].predict_proba(features)[:, 1]
    else:
        probs = [model.predict_proba(features)[:, 1] for model in bundle["models"].values()]
        target_prob = np.mean(probs, axis=0)
    return (
        target_prob,
        int(bundle.get("prediction_background_count", bundle.get("background_per_subject_session", 90))),
        float(bundle.get("decision_threshold", CONFIG["decision_threshold"])),
        selected_model,
        bundle.get("prediction_rule", CONFIG["prediction_rule"]),
    )


def predict_with_neural_model(test_df: pd.DataFrame) -> np.ndarray:
    if torch is None or DataLoader is None or EEGDataset is None or Net is None:
        raise RuntimeError("PyTorch is required for neural fallback inference.")

    device = get_device(CONFIG["use_cpu"])
    test_dataset = EEGDataset(data_dir=CONFIG["test_dir"], label_csv=None, normalize=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    sample_eeg, _ = test_dataset[0]
    checkpoint = torch.load(CONFIG["neural_model_path"], map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model = Net(input_shape=tuple(sample_eeg.shape)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    file_to_prob = {}
    with torch.no_grad():
        for eeg, eeg_files in test_loader:
            logits = model(eeg.to(device, non_blocking=True))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for eeg_file, value in zip(eeg_files, prob):
                file_to_prob[eeg_file] = float(value)
    return np.asarray([file_to_prob[name] for name in test_df["eeg_file"]]), int(
        checkpoint.get("background_per_subject_session", 90) if isinstance(checkpoint, dict) else 90
    )


def main() -> None:
    test_files = sorted(path.name for path in CONFIG["test_dir"].glob("*.npy"))
    test_df = add_metadata(pd.DataFrame({"eeg_file": test_files}))

    if CONFIG["tabular_model_path"].exists():
        target_prob, background_count, threshold, model_name, prediction_rule = predict_with_tabular_model(test_df)
    elif CONFIG["use_neural_fallback"]:
        target_prob, background_count = predict_with_neural_model(test_df)
        threshold = CONFIG["decision_threshold"]
        model_name = "neural_fallback"
        prediction_rule = CONFIG["prediction_rule"]
    else:
        raise FileNotFoundError(
            f"Missing {CONFIG['tabular_model_path']}. Run python train.py before python test.py."
        )

    if prediction_rule == "fixed_background":
        predictions = fixed_background_predictions(test_df, target_prob, background_count)
    else:
        predictions = (target_prob >= threshold).astype(int)
    output = pd.DataFrame(
        {
            "eeg_file": test_df["eeg_file"],
            "prediction": [INDEX_TO_LABEL[int(value)] for value in predictions],
        }
    )
    probability_output = pd.DataFrame(
        {
            "eeg_file": test_df["eeg_file"],
            "prob_target": target_prob,
            "prediction": output["prediction"],
            "model": model_name,
            "threshold": threshold,
            "prediction_rule": prediction_rule,
            "background_count": background_count,
        }
    )

    RES_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(CONFIG["output_csv"], index=False)
    probability_output.to_csv(CONFIG["probability_csv"], index=False)
    print(f"Predictions saved to: {CONFIG['output_csv']}")
    print(output["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()
