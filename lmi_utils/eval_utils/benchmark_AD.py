import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

from anomaly_detectors.anomalib_lmi.v1.model import AnomalyModel as AnomalyModelV1
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

logger = logging.getLogger(__name__)


def get_average_of_top_k(anomaly_map: np.ndarray, top_k: Union[float, int]) -> float:
    """
    Calculates the mean of the top k scores in an anomaly map.

    Args:
        anomaly_map (np.array): A 2D numpy array (e.g., 224x224).
        top_k (float or int):
            - If float (0.0 to 1.0): Treated as a ratio (e.g., 0.1 = top 10%).
            - If int (> 1): Treated as an exact number of pixels (e.g., 100 = top 100 pixels).

    Returns:
        float: The average score of the selected pixels.
    """
    flat_scores = anomaly_map.flatten()
    num_pixels = flat_scores.size

    if isinstance(top_k, float) and 0 < top_k <= 1.0:
        k = int(num_pixels * top_k)
    elif isinstance(top_k, int) and top_k >= 1:
        k = top_k
    else:
        # Fallback for edge cases (like 0)
        k = 1

    # Safety check: Ensure k is within valid bounds [1, num_pixels]
    k = max(1, min(k, num_pixels))

    # Partition to find the top k values
    top_k_values = np.partition(flat_scores, -k)[-k:]
    return np.mean(top_k_values)


def get_optimal_threshold(
    y_true: Union[List[int], np.ndarray], y_scores: Union[List[float], np.ndarray], thresholds: np.ndarray
) -> Tuple[float, float]:
    """
    Calculates F1-Max for a single model.
    """
    best_f1 = 0
    best_threshold = 0

    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = th
    return best_threshold, best_f1


def plot_roc_curve(
    y_true: Union[List[int], np.ndarray], y_scores: Union[List[float], np.ndarray], model_name: str, axes: Any
) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    optimal_threshold, optimal_f1 = get_optimal_threshold(y_true, y_scores, thresholds)

    # Plot the curve
    axes.plot(
        fpr,
        tpr,
        marker=".",
        label=f"{model_name} (AUC = {roc_auc:.2f}, Optimal Threshold = {optimal_threshold:.2f} with F1 = {optimal_f1:.2f})",
    )
    return optimal_threshold, optimal_f1


def load_preprocess(json_path: str) -> Optional[List[Dict]]:
    """
    Load preprocess configuration from json file.

    Returns:
        List of preprocessing operations.
    """

    def parse(data: List[Dict[str, Any]]) -> List[Dict]:
        supported_types = {"resize", "tile"}
        tiling_keys = {"height", "width", "xStride", "yStride"}

        ops = []
        for preprocess in data:
            if "type" not in preprocess or "configuration" not in preprocess:
                raise ValueError("Must contain 'type' and 'configuration' keys.")

            p_type = preprocess["type"]
            config = preprocess["configuration"]

            if p_type not in supported_types:
                raise ValueError(f"Unsupported type '{p_type}'.")

            if p_type == "tile":
                if not tiling_keys.issubset(config.keys()):
                    raise ValueError(f"Tiling configuration must contain keys: {tiling_keys}.")

                # Format the tile config immediately
                tile_config = {"tile_size": [config["height"], config["width"]], "stride": [config["yStride"], config["xStride"]]}
                ops.append({"type": "tile", "configuration": tile_config})
            elif p_type == "resize":
                ops.append(preprocess)
        return ops

    # --- Main Execution ---
    with open(json_path, "r") as f:
        data = json.load(f)
    return parse(data)


def horizontal_stack(
    images: List[np.ndarray], padding_width: int = 20, header_height: int = 50, bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
    """
    Stacks a list of images horizontally with padding.

    Args:
        images: List of numpy arrays (images).
        padding_width: Int, pixels between images.
        header_height: Int, pixels for top padding.
        bg_color: Tuple (B, G, R) for background color. Default black.
    """
    if not images:
        return None

    # 1. Standardize images (Handle Grayscale)
    processed_imgs = []
    for img in images:
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        processed_imgs.append(img)

    # 2. Calculate Final Dimensions (max height, total width + paddings)
    final_h = max(img.shape[0] for img in processed_imgs) + header_height
    total_img_width = sum(img.shape[1] for img in processed_imgs)
    total_padding = padding_width * (len(processed_imgs) - 1)
    final_w = total_img_width + total_padding

    # 3. Create Canvas
    out = np.full((final_h, final_w, 3), bg_color, dtype=np.uint8)

    # 4. Place Images
    current_x = 0
    anchors = []
    for img in processed_imgs:
        h, w, _ = img.shape
        out[header_height : header_height + h, current_x : current_x + w] = img
        text_pos = (current_x, int(header_height / 1.5))
        anchors.append(text_pos)
        current_x += w + padding_width
    return out, anchors


def load_and_validate_data(data_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Load and validate image data from good and bad subdirectories.

    Args:
        data_path: Path to data directory containing 'good' and 'bad' subdirectories.

    Returns:
        Tuple of (good_list, bad_list) containing paths to images.
    """
    good_path = data_path / "good"
    bad_path = data_path / "bad"
    if not good_path.is_dir() or not bad_path.is_dir():
        raise ValueError(f"Data directory must contain 'good' and 'bad' subdirectories. Given: {data_path}")

    good_list = list(good_path.glob("*.jpg"))
    bad_list = list(bad_path.glob("*.jpg"))
    return good_list, bad_list


def load_models(model1_path: Path, model2_path: Path) -> Tuple[AnomalyModelV1, AnomalyModelV1]:
    """
    Load two anomaly detection models.

    Args:
        model1_path: Path to first model.
        model2_path: Path to second model.

    Returns:
        Tuple of (model1, model2).
    """
    model1 = AnomalyModelV1(model1_path)
    model2 = AnomalyModelV1(model2_path)
    return model1, model2


def process_single_image(
    image: np.ndarray,
    model1: AnomalyModelV1,
    model2: AnomalyModelV1,
    ops1: List[Dict],
    ops2: List[Dict],
    preprocessor: Preprocessor,
    reconstructor: Reconstructor,
    topk: Optional[float] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single image with both models.

    Args:
        image: Input image (RGB).
        model1: First anomaly detection model.
        model2: Second anomaly detection model.
        ops1: Preprocessing operations for model1.
        ops2: Preprocessing operations for model2.
        preprocessor: Preprocessor instance.
        reconstructor: Reconstructor instance.
        topk: Optional top k ratio or number for scoring. If None, only reconstructed data is returned.

    Returns:
        Tuple of (score1, score2, scores1_recon, scores2_recon, im1_recon, im2_recon).
        If topk is None, score1 and score2 will be None.
    """
    # Model 1 processing
    imgs1, ops1_recon = preprocessor.preprocess(image, ops1)
    scores1 = [model1.predict(i1) for i1 in imgs1]
    scores1_recon = reconstructor.reconstruct(scores1, ops1_recon)[0]
    im1_recon = reconstructor.reconstruct(imgs1, ops1_recon)[0]

    # Model 2 processing
    imgs2, ops2_recon = preprocessor.preprocess(image, ops2)
    scores2 = [model2.predict(i2) for i2 in imgs2]
    scores2_recon = reconstructor.reconstruct(scores2, ops2_recon)[0]
    im2_recon = reconstructor.reconstruct(imgs2, ops2_recon)[0]

    # Calculate aggregate scores if topk is provided
    score1 = get_average_of_top_k(scores1_recon, topk) if topk is not None else None
    score2 = get_average_of_top_k(scores2_recon, topk) if topk is not None else None

    return score1, score2, scores1_recon, scores2_recon, im1_recon, im2_recon


def run_inference(
    image_paths: List[Path], model1: AnomalyModelV1, model2: AnomalyModelV1, ops1: List[Dict], ops2: List[Dict], topk: float
) -> Tuple[List[float], List[float]]:
    """
    Run inference on images using both models and calculate anomaly scores.

    Args:
        image_paths: List of image paths to process.
        model1: First anomaly detection model.
        model2: Second anomaly detection model.
        ops1: Preprocessing operations for model1.
        ops2: Preprocessing operations for model2.
        topk: Top k ratio or number for scoring.

    Returns:
        Tuple of (model1_scores, model2_scores).
    """
    model1_scores = []
    model2_scores = []
    preprocessor = Preprocessor()
    reconstructor = Reconstructor()

    logger.info(f"Running inference on {len(image_paths)} images...")
    for img_path in image_paths:
        im = cv2.imread(str(img_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        score1, score2, _, _, _, _ = process_single_image(im, model1, model2, ops1, ops2, preprocessor, reconstructor, topk)
        model1_scores.append(score1)
        model2_scores.append(score2)

    return model1_scores, model2_scores


def plot_and_save_roc(
    y_true: List[int], model1_scores: List[float], model2_scores: List[float], output_dir: Path
) -> Tuple[float, float, float, float]:
    """
    Plot ROC curves for both models and save to file.

    Args:
        y_true: Ground truth labels.
        model1_scores: Anomaly scores from model1.
        model2_scores: Anomaly scores from model2.
        output_dir: Directory to save the plot.

    Returns:
        Tuple of (optimal_threshold1, optimal_f1_1, optimal_threshold2, optimal_f1_2).
    """
    plt.figure(figsize=(10, 6))
    optimal_threshold1, optimal_f1_1 = plot_roc_curve(y_true, model1_scores, "model1", plt)
    optimal_threshold2, optimal_f1_2 = plot_roc_curve(y_true, model2_scores, "model2", plt)

    # Plot formatting
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate (False Alarms)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Anomaly Detection Benchmark: model1 vs model2")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save the image
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "ad_benchmark.png")

    return optimal_threshold1, optimal_f1_1, optimal_threshold2, optimal_f1_2


def annotate_and_save_images(
    image_paths: List[Path],
    good_list: List[Path],
    model1: AnomalyModelV1,
    model2: AnomalyModelV1,
    model1_scores: List[float],
    model2_scores: List[float],
    ops1: List[Dict],
    ops2: List[Dict],
    optimal_threshold1: float,
    optimal_threshold2: float,
    output_dir: Path,
) -> None:
    """
    Annotate images with model predictions and save to output directory.

    Args:
        image_paths: List of all image paths.
        good_list: List of good image paths (for determining FP/FN).
        model1: First anomaly detection model.
        model2: Second anomaly detection model.
        model1_scores: Anomaly scores from model1.
        model2_scores: Anomaly scores from model2.
        ops1: Preprocessing operations for model1.
        ops2: Preprocessing operations for model2.
        optimal_threshold1: Optimal threshold for model1.
        optimal_threshold2: Optimal threshold for model2.
        output_dir: Directory to save annotated images.
    """
    preprocessor = Preprocessor()
    reconstructor = Reconstructor()

    logger.info("Annotating images...")
    for j, img_path in enumerate(image_paths):
        im = cv2.imread(str(img_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Process image with both models
        _, _, scores1_recon, scores2_recon, im1_recon, im2_recon = process_single_image(
            im, model1, model2, ops1, ops2, preprocessor, reconstructor
        )

        # Annotate images
        annotated_img1 = model1.annotate(im1_recon, scores1_recon, optimal_threshold1, scores1_recon.max())
        annotated_img2 = model2.annotate(im2_recon, scores2_recon, optimal_threshold2, scores2_recon.max())
        annotated_img, anchors = horizontal_stack([im1_recon, annotated_img1, annotated_img2])

        # Add decision text
        model1_decision = "PASS" if model1_scores[j] < optimal_threshold1 else "FAIL"
        model2_decision = "PASS" if model2_scores[j] < optimal_threshold2 else "FAIL"
        cv2.putText(annotated_img, f"model1 {model1_decision}", anchors[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_img, f"model2 {model2_decision}", anchors[2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Calculate FP/FN
        FP_model1 = 1 if model1_decision == "FAIL" and j < len(good_list) else 0
        FP_model2 = 1 if model2_decision == "FAIL" and j < len(good_list) else 0
        FN_model1 = 1 if model1_decision == "PASS" and j >= len(good_list) else 0
        FN_model2 = 1 if model2_decision == "PASS" and j >= len(good_list) else 0
        FP = FP_model1 + FP_model2
        FN = FN_model1 + FN_model2

        # Save annotated image
        outp = output_dir / "good" if j < len(good_list) else output_dir / "bad"
        outp.mkdir(exist_ok=True)
        out_name = f"{img_path.name.split('.')[0]}_FP-{FP}_FN-{FN}.jpg"
        bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outp / out_name), bgr)


def save_metrics_to_csv(
    y_true: List[int],
    model1_scores: List[float],
    model2_scores: List[float],
    optimal_threshold1: float,
    optimal_threshold2: float,
    optimal_f1_1: float,
    optimal_f1_2: float,
    output_dir: Path,
) -> None:
    """
    Calculate and save metrics to CSV file.

    Args:
        y_true: Ground truth labels.
        model1_scores: Anomaly scores from model1.
        model2_scores: Anomaly scores from model2.
        optimal_threshold1: Optimal threshold for model1.
        optimal_threshold2: Optimal threshold for model2.
        optimal_f1_1: Optimal F1 score for model1.
        optimal_f1_2: Optimal F1 score for model2.
        output_dir: Directory to save the CSV file.
    """
    # Calculate predictions at optimal thresholds
    model1_preds = [1 if s >= optimal_threshold1 else 0 for s in model1_scores]
    model2_preds = [1 if s >= optimal_threshold2 else 0 for s in model2_scores]

    # Calculate precision and recall
    p1 = precision_score(y_true, model1_preds)
    r1 = recall_score(y_true, model1_preds)
    p2 = precision_score(y_true, model2_preds)
    r2 = recall_score(y_true, model2_preds)

    # Save metrics to CSV
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Optimal Threshold", "F1", "Precision", "Recall"])
        writer.writerow(["model1", f"{optimal_threshold1:.4f}", f"{optimal_f1_1:.4f}", f"{p1:.4f}", f"{r1:.4f}"])
        writer.writerow(["model2", f"{optimal_threshold2:.4f}", f"{optimal_f1_2:.4f}", f"{p2:.4f}", f"{r2:.4f}"])

    logger.info(f"Metrics saved to {csv_file}")


def main() -> None:
    """Main function to run anomaly detection benchmark."""
    logging.basicConfig(level=logging.INFO)
    # Parse arguments
    ap = argparse.ArgumentParser("Anomaly Detection Benchmark: Compare two models on the same dataset")
    ap.add_argument("--data", required=True, type=Path, help='Path to data directory, which must contain "good" and "bad" subdirectories')
    ap.add_argument("--model1", required=True, type=Path, help="Path to first model")
    ap.add_argument("--model2", required=True, type=Path, help="Path to second model")
    ap.add_argument(
        "--preprocess1",
        required=True,
        type=Path,
        help='Path to first model preprocessing.json. Can be found in "training_input" in GCP model bucket.',
    )
    ap.add_argument(
        "--preprocess2",
        required=True,
        type=Path,
        help='Path to second model preprocessing.json. Can be found in "training_input" in GCP model bucket.',
    )
    ap.add_argument("--topk", default=0.1, type=float, help="Top k ratio or number for scoring. default=0.1 (top 10%%)")
    ap.add_argument("--output-dir", default=Path("./output"), type=Path, help="Directory to save outputs")
    args = ap.parse_args()

    # Load and validate data
    good_list, bad_list = load_and_validate_data(args.data)

    # Load preprocessing configurations
    ops1 = load_preprocess(args.preprocess1)
    ops2 = load_preprocess(args.preprocess2)

    # Create output directory
    output_dir = args.output_dir / datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

    # Load models
    model1, model2 = load_models(args.model1, args.model2)

    # Prepare data
    all_images = good_list + bad_list
    y_true = [0] * len(good_list) + [1] * len(bad_list)

    # Run inference
    model1_scores, model2_scores = run_inference(all_images, model1, model2, ops1, ops2, args.topk)

    # Plot and save ROC curves
    optimal_threshold1, optimal_f1_1, optimal_threshold2, optimal_f1_2 = plot_and_save_roc(y_true, model1_scores, model2_scores, output_dir)

    # Annotate and save images
    annotate_and_save_images(
        all_images, good_list, model1, model2, model1_scores, model2_scores, ops1, ops2, optimal_threshold1, optimal_threshold2, output_dir
    )

    # Save metrics to CSV
    save_metrics_to_csv(
        y_true, model1_scores, model2_scores, optimal_threshold1, optimal_threshold2, optimal_f1_1, optimal_f1_2, output_dir
    )


if __name__ == "__main__":
    main()
