from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.core.preprocessing import prepare_mobilenet_input
from app.core.xai import (
    XAI_CANDIDATE_LAYERS,
    _make_gradcam_pp_heatmap,
    load_gradcam_models,
)


SEED = 42
BOOTSTRAP_REPETITIONS = 10_000

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "MVTec_AD_DATASET"
    / "bottle"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

GRADCAM_PER_IMAGE_PATH = (
    RESULTS_DIR
    / "xai_localization_validation_per_image.csv"
)

GRADCAM_SUMMARY_PATH = (
    RESULTS_DIR
    / "xai_localization_validation_summary.csv"
)

GRADCAM_SELECTION_PATH = (
    RESULTS_DIR
    / "xai_localization_validation_selection.json"
)

METHOD_PER_IMAGE_PATH = (
    RESULTS_DIR
    / "xai_method_validation_per_image.csv"
)

METHOD_SUMMARY_PATH = (
    RESULTS_DIR
    / "xai_method_validation_summary.csv"
)

METHOD_SELECTION_PATH = (
    RESULTS_DIR
    / "xai_method_validation_selection.json"
)

DEFECT_CLASSES = (
    "broken_large",
    "broken_small",
    "contamination",
)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def build_source_dataframe() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for source_partition in ("train", "test"):
        for source_class in (
            "good",
            *DEFECT_CLASSES,
        ):
            folder = (
                DATASET_ROOT
                / source_partition
                / source_class
            )

            if not folder.is_dir():
                continue

            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                if "_mask" in path.name.lower():
                    continue

                rows.append(
                    {
                        "filepath": str(path),
                        "filename": path.name,
                        "label": (
                            "good"
                            if source_class == "good"
                            else "defective"
                        ),
                        "source_partition": source_partition,
                        "source_class": source_class,
                        "source_stratum": (
                            f"{source_partition}:{source_class}"
                        ),
                    }
                )

    dataframe = pd.DataFrame(rows)

    if len(dataframe) != 292:
        raise RuntimeError(
            "Unexpected MVTec bottle volume. "
            f"Expected 292 images, found {len(dataframe)}."
        )

    return dataframe


def reproduce_validation_split(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=0.40,
        stratify=dataframe["source_stratum"],
        random_state=SEED,
    )

    development_df, test_df = train_test_split(
        holdout_df,
        test_size=0.15 / 0.40,
        stratify=holdout_df["source_stratum"],
        random_state=SEED,
    )

    val_df, calibration_df = train_test_split(
        development_df,
        test_size=0.10 / 0.25,
        stratify=development_df["source_stratum"],
        random_state=SEED,
    )

    observed = (
        len(train_df),
        len(val_df),
        len(calibration_df),
        len(test_df),
    )

    expected = (
        175,
        43,
        30,
        44,
    )

    if observed != expected:
        raise RuntimeError(
            "Notebook 4 split reproduction failed. "
            f"Expected {expected}, observed {observed}."
        )

    defective_validation = val_df[
        val_df["source_class"].isin(
            DEFECT_CLASSES
        )
    ].copy()

    defective_validation = (
        defective_validation.sort_values(
            [
                "source_class",
                "filename",
            ]
        )
        .reset_index(drop=True)
    )

    if len(defective_validation) != 9:
        raise RuntimeError(
            "Unexpected Validation defective volume. "
            f"Expected 9, found {len(defective_validation)}."
        )

    return defective_validation


def resolve_mask_path(
    source_class: str,
    filename: str,
) -> Path:
    stem = Path(filename).stem

    mask_path = (
        DATASET_ROOT
        / "ground_truth"
        / source_class
        / f"{stem}_mask.png"
    )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth mask: {mask_path}"
        )

    return mask_path


def load_binary_mask(
    mask_path: Path,
) -> np.ndarray:
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(
            mask_image.convert("L"),
            dtype=np.uint8,
        )

    binary = mask > 0

    if not binary.any():
        raise RuntimeError(
            f"Ground-truth mask is empty: {mask_path}"
        )

    return binary


def resize_heatmap(
    heatmap: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    height, width = target_shape

    heatmap_image = Image.fromarray(
        heatmap.astype(
            np.float32
        )
    )

    resized = heatmap_image.resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )

    result = np.asarray(
        resized,
        dtype=np.float32,
    )

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Resized XAI heatmap contains non-finite values."
        )

    return result


def calculate_localization_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    mask_flat = mask.astype(
        np.uint8
    ).reshape(-1)

    heatmap_flat = heatmap.reshape(-1)

    pixel_ap = float(
        average_precision_score(
            mask_flat,
            heatmap_flat,
        )
    )

    pixel_roc_auc = float(
        roc_auc_score(
            mask_flat,
            heatmap_flat,
        )
    )

    heatmap_energy = float(
        heatmap.sum()
    )

    if heatmap_energy <= 1e-12:
        raise RuntimeError(
            "Heatmap contains no usable localization energy."
        )

    energy_inside_mask = float(
        heatmap[mask].sum()
        / heatmap_energy
    )

    mask_fraction = float(
        mask.mean()
    )

    energy_gain = float(
        energy_inside_mask
        / mask_fraction
    )

    mask_float = mask.astype(
        np.float32
    )

    intersection = float(
        np.sum(
            heatmap
            * mask_float
        )
    )

    union = float(
        np.sum(heatmap)
        + np.sum(mask_float)
        - intersection
    )

    soft_iou = (
        intersection / union
        if union > 0.0
        else 0.0
    )

    peak_index = np.unravel_index(
        int(np.argmax(heatmap)),
        heatmap.shape,
    )

    pointing_hit = float(
        mask[peak_index]
    )

    return {
        "pixel_ap": pixel_ap,
        "pixel_roc_auc": pixel_roc_auc,
        "energy_inside_mask": energy_inside_mask,
        "mask_fraction": mask_fraction,
        "energy_gain": energy_gain,
        "soft_iou": float(soft_iou),
        "pointing_hit": pointing_hit,
    }


def load_locked_gradcam_baseline(
    validation_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    required_paths = (
        GRADCAM_PER_IMAGE_PATH,
        GRADCAM_SUMMARY_PATH,
        GRADCAM_SELECTION_PATH,
    )

    missing = [
        str(path)
        for path in required_paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Locked Grad-CAM layer-selection artifacts are missing:\n"
            + "\n".join(missing)
        )

    selection = json.loads(
        GRADCAM_SELECTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    if selection.get(
        "locked_test_used_for_selection"
    ) is not False:
        raise RuntimeError(
            "Invalid Grad-CAM selection contract: "
            "locked Test must not be used for XAI selection."
        )

    if selection.get(
        "validation_defective_images"
    ) != 9:
        raise RuntimeError(
            "Invalid Grad-CAM selection contract: "
            "expected 9 Validation defective images."
        )

    selected_layer = str(
        selection.get(
            "selected_layer",
            "",
        )
    )

    if selected_layer not in XAI_CANDIDATE_LAYERS:
        raise RuntimeError(
            "Invalid or unsupported previously selected "
            f"Grad-CAM layer: {selected_layer!r}."
        )

    summary_df = pd.read_csv(
        GRADCAM_SUMMARY_PATH
    )

    if summary_df.empty:
        raise RuntimeError(
            "Grad-CAM layer-selection summary is empty."
        )

    summary_winner = str(
        summary_df.iloc[0]["layer"]
    )

    if summary_winner != selected_layer:
        raise RuntimeError(
            "Grad-CAM selection artifact disagrees "
            "with the summary ranking."
        )

    baseline_df = pd.read_csv(
        GRADCAM_PER_IMAGE_PATH
    )

    baseline_df = baseline_df[
        baseline_df["layer"] == selected_layer
    ].copy()

    if len(baseline_df) != 9:
        raise RuntimeError(
            "Expected exactly 9 locked Grad-CAM "
            f"Validation rows for '{selected_layer}', "
            f"found {len(baseline_df)}."
        )

    expected_keys = set(
        validation_df[
            [
                "source_class",
                "filename",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    observed_keys = set(
        baseline_df[
            [
                "source_class",
                "filename",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if observed_keys != expected_keys:
        raise RuntimeError(
            "Locked Grad-CAM baseline does not match "
            "the reconstructed Validation sample set."
        )

    baseline_df.insert(
        0,
        "method",
        "gradcam",
    )

    return (
        selected_layer,
        baseline_df,
    )


def evaluate_gradcam_pp(
    validation_df: pd.DataFrame,
    selected_layer: str,
) -> pd.DataFrame:
    models = load_gradcam_models(
        selected_layer
    )

    records: list[dict[str, object]] = []

    print()
    print(
        "Evaluating Grad-CAM++ @ "
        f"{selected_layer} "
        f"{models.conv_model.output_shape}"
    )

    for _, row in validation_df.iterrows():
        image_path = Path(
            row["filepath"]
        )

        mask_path = resolve_mask_path(
            source_class=row["source_class"],
            filename=row["filename"],
        )

        with Image.open(image_path) as source_image:
            image = source_image.convert(
                "RGB"
            )

        mask = load_binary_mask(
            mask_path
        )

        image_batch = prepare_mobilenet_input(
            image
        )

        heatmap = _make_gradcam_pp_heatmap(
            image_batch,
            models,
            target_class=0,
        )

        resized_heatmap = resize_heatmap(
            heatmap,
            mask.shape,
        )

        metrics = calculate_localization_metrics(
            resized_heatmap,
            mask,
        )

        records.append(
            {
                "method": "gradcam_pp",
                "layer": selected_layer,
                "heatmap_height": int(
                    heatmap.shape[0]
                ),
                "heatmap_width": int(
                    heatmap.shape[1]
                ),
                "source_class": row[
                    "source_class"
                ],
                "filename": row[
                    "filename"
                ],
                **metrics,
            }
        )

        print(
            f"  {row['source_class']:14s} "
            f"{row['filename']:8s} "
            f"AP={metrics['pixel_ap']:.4f} "
            f"SoftIoU={metrics['soft_iou']:.4f} "
            f"Point={int(metrics['pointing_hit'])}"
        )

    result = pd.DataFrame(
        records
    )

    if len(result) != 9:
        raise RuntimeError(
            "Grad-CAM++ evaluation did not produce "
            "exactly 9 Validation results."
        )

    return result


def build_method_summary(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_df = (
        combined_df.groupby(
            [
                "method",
                "layer",
                "heatmap_height",
                "heatmap_width",
            ],
            as_index=False,
        )
        .agg(
            sample_count=(
                "pixel_ap",
                "count",
            ),
            mean_pixel_ap=(
                "pixel_ap",
                "mean",
            ),
            std_pixel_ap=(
                "pixel_ap",
                "std",
            ),
            mean_pixel_roc_auc=(
                "pixel_roc_auc",
                "mean",
            ),
            mean_soft_iou=(
                "soft_iou",
                "mean",
            ),
            mean_energy_inside_mask=(
                "energy_inside_mask",
                "mean",
            ),
            mean_energy_gain=(
                "energy_gain",
                "mean",
            ),
            pointing_accuracy=(
                "pointing_hit",
                "mean",
            ),
        )
    )

    return (
        summary_df.sort_values(
            by=[
                "mean_pixel_ap",
                "mean_soft_iou",
                "pointing_accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def paired_bootstrap_delta(
    combined_df: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    pivot = combined_df.pivot(
        index=[
            "source_class",
            "filename",
        ],
        columns="method",
        values=metric,
    )

    required_methods = {
        "gradcam",
        "gradcam_pp",
    }

    if set(pivot.columns) != required_methods:
        raise RuntimeError(
            f"Paired bootstrap for '{metric}' "
            "does not contain both XAI methods."
        )

    if len(pivot) != 9:
        raise RuntimeError(
            f"Paired bootstrap for '{metric}' "
            f"expected 9 samples, found {len(pivot)}."
        )

    deltas = (
        pivot["gradcam_pp"]
        - pivot["gradcam"]
    ).to_numpy(
        dtype=np.float64
    )

    rng = np.random.default_rng(
        SEED
    )

    bootstrap_indices = rng.integers(
        low=0,
        high=len(deltas),
        size=(
            BOOTSTRAP_REPETITIONS,
            len(deltas),
        ),
    )

    bootstrap_means = deltas[
        bootstrap_indices
    ].mean(
        axis=1
    )

    lower, upper = np.percentile(
        bootstrap_means,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "mean_delta_gradcam_pp_minus_gradcam": float(
            deltas.mean()
        ),
        "median_delta_gradcam_pp_minus_gradcam": float(
            np.median(deltas)
        ),
        "bootstrap_95_ci_lower": float(
            lower
        ),
        "bootstrap_95_ci_upper": float(
            upper
        ),
        "gradcam_pp_sample_wins": int(
            np.sum(
                deltas > 0.0
            )
        ),
        "gradcam_sample_wins": int(
            np.sum(
                deltas < 0.0
            )
        ),
        "ties": int(
            np.sum(
                deltas == 0.0
            )
        ),
    }


def run_benchmark() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = build_source_dataframe()

    validation_df = reproduce_validation_split(
        dataframe
    )

    (
        selected_layer,
        gradcam_baseline_df,
    ) = load_locked_gradcam_baseline(
        validation_df
    )

    print()
    print("=" * 88)
    print("LOCKED GRAD-CAM BASELINE")
    print("=" * 88)
    print(
        "Validation-selected layer:",
        selected_layer,
    )
    print(
        "Baseline rows:",
        len(gradcam_baseline_df),
    )
    print(
        "Locked Test used for selection:",
        False,
    )

    gradcam_pp_df = evaluate_gradcam_pp(
        validation_df,
        selected_layer,
    )

    combined_df = pd.concat(
        [
            gradcam_baseline_df,
            gradcam_pp_df,
        ],
        ignore_index=True,
        sort=False,
    )

    method_summary_df = build_method_summary(
        combined_df
    )

    pixel_ap_bootstrap = paired_bootstrap_delta(
        combined_df,
        "pixel_ap",
    )

    soft_iou_bootstrap = paired_bootstrap_delta(
        combined_df,
        "soft_iou",
    )

    selected_method = str(
        method_summary_df.iloc[0][
            "method"
        ]
    )

    selected_method_layer = str(
        method_summary_df.iloc[0][
            "layer"
        ]
    )

    combined_df.to_csv(
        METHOD_PER_IMAGE_PATH,
        index=False,
    )

    method_summary_df.to_csv(
        METHOD_SUMMARY_PATH,
        index=False,
    )

    method_selection = {
        "protocol": (
            "Notebook 4 Validation-only "
            "ground-truth-mask XAI method selection"
        ),
        "seed": SEED,
        "validation_defective_images": int(
            len(validation_df)
        ),
        "layer_selection_source": (
            GRADCAM_SELECTION_PATH.name
        ),
        "locked_validation_selected_layer": (
            selected_layer
        ),
        "candidate_methods": [
            "gradcam",
            "gradcam_pp",
        ],
        "primary_selection_metric": (
            "mean_pixel_ap"
        ),
        "secondary_selection_metric": (
            "mean_soft_iou"
        ),
        "tertiary_selection_metric": (
            "pointing_accuracy"
        ),
        "selected_method": selected_method,
        "selected_layer": selected_method_layer,
        "pixel_ap_paired_bootstrap": (
            pixel_ap_bootstrap
        ),
        "soft_iou_paired_bootstrap": (
            soft_iou_bootstrap
        ),
        "bootstrap_repetitions": (
            BOOTSTRAP_REPETITIONS
        ),
        "locked_test_used_for_selection": False,
    }

    METHOD_SELECTION_PATH.write_text(
        json.dumps(
            method_selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("VALIDATION XAI METHOD SUMMARY")
    print("=" * 88)
    print(
        method_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 88)
    print("PAIRED VALIDATION BOOTSTRAP")
    print("=" * 88)

    print(
        "Pixel AP delta "
        "(Grad-CAM++ - Grad-CAM): "
        f"{pixel_ap_bootstrap['mean_delta_gradcam_pp_minus_gradcam']:.6f}"
    )

    print(
        "Pixel AP 95% bootstrap CI: "
        f"[{pixel_ap_bootstrap['bootstrap_95_ci_lower']:.6f}, "
        f"{pixel_ap_bootstrap['bootstrap_95_ci_upper']:.6f}]"
    )

    print(
        "Pixel AP sample wins "
        f"(PP / CAM / ties): "
        f"{pixel_ap_bootstrap['gradcam_pp_sample_wins']} / "
        f"{pixel_ap_bootstrap['gradcam_sample_wins']} / "
        f"{pixel_ap_bootstrap['ties']}"
    )

    print()
    print(
        "Soft-IoU delta "
        "(Grad-CAM++ - Grad-CAM): "
        f"{soft_iou_bootstrap['mean_delta_gradcam_pp_minus_gradcam']:.6f}"
    )

    print(
        "Soft-IoU 95% bootstrap CI: "
        f"[{soft_iou_bootstrap['bootstrap_95_ci_lower']:.6f}, "
        f"{soft_iou_bootstrap['bootstrap_95_ci_upper']:.6f}]"
    )

    print()
    print(
        "VALIDATION-SELECTED METHOD:",
        selected_method,
    )

    print(
        "VALIDATION-SELECTED LAYER:",
        selected_method_layer,
    )

    print()
    print(
        "Saved:",
        METHOD_PER_IMAGE_PATH,
    )

    print(
        "Saved:",
        METHOD_SUMMARY_PATH,
    )

    print(
        "Saved:",
        METHOD_SELECTION_PATH,
    )


if __name__ == "__main__":
    run_benchmark()