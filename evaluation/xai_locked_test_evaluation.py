from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from app.core.preprocessing import prepare_mobilenet_input
from app.core.xai import (
    _make_gradcam_heatmap,
    load_gradcam_models,
)
from evaluation.xai_localization_benchmark import (
    DEFECT_CLASSES,
    METHOD_SELECTION_PATH,
    RESULTS_DIR,
    SEED,
    build_source_dataframe,
    calculate_localization_metrics,
    load_binary_mask,
    resize_heatmap,
    resolve_mask_path,
)


BOOTSTRAP_REPETITIONS = 10_000

TEST_PER_IMAGE_PATH = (
    RESULTS_DIR
    / "xai_locked_test_per_image.csv"
)

TEST_SUMMARY_PATH = (
    RESULTS_DIR
    / "xai_locked_test_summary.csv"
)

TEST_CLASS_SUMMARY_PATH = (
    RESULTS_DIR
    / "xai_locked_test_per_defect_class.csv"
)

TEST_REPORT_PATH = (
    RESULTS_DIR
    / "xai_locked_test_report.json"
)


def reproduce_locked_test_split(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 4's deterministic 60/15/10/15 split.

    The returned Test partition is the same locked custom benchmark
    defined by Notebook 4. No Test information is used for XAI method
    or layer selection.
    """
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
            "Notebook 4 locked split reproduction failed. "
            f"Expected {expected}, observed {observed}."
        )

    defective_test = test_df[
        test_df["source_class"].isin(
            DEFECT_CLASSES
        )
    ].copy()

    defective_test = (
        defective_test.sort_values(
            [
                "source_class",
                "filename",
            ]
        )
        .reset_index(drop=True)
    )

    if defective_test.empty:
        raise RuntimeError(
            "Locked Test contains no defective samples."
        )

    observed_classes = set(
        defective_test["source_class"]
    )

    expected_classes = set(
        DEFECT_CLASSES
    )

    if observed_classes != expected_classes:
        raise RuntimeError(
            "Locked Test does not contain all expected "
            f"defect classes. Observed: {sorted(observed_classes)}."
        )

    return defective_test


def load_frozen_xai_selection() -> tuple[str, str]:
    """
    Load the Validation-only XAI selection and freeze it.

    Test evaluation is forbidden from changing this selection.
    """
    if not METHOD_SELECTION_PATH.is_file():
        raise FileNotFoundError(
            "Validation XAI method-selection artifact is missing: "
            f"{METHOD_SELECTION_PATH}"
        )

    selection = json.loads(
        METHOD_SELECTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    if selection.get(
        "locked_test_used_for_selection"
    ) is not False:
        raise RuntimeError(
            "Invalid XAI selection contract: "
            "locked Test was marked as used during selection."
        )

    method = str(
        selection.get(
            "selected_method",
            "",
        )
    )

    layer = str(
        selection.get(
            "selected_layer",
            "",
        )
    )

    if method != "gradcam":
        raise RuntimeError(
            "Unexpected frozen XAI method. "
            f"Expected 'gradcam', found {method!r}."
        )

    if layer != "out_relu":
        raise RuntimeError(
            "Unexpected frozen XAI layer. "
            f"Expected 'out_relu', found {layer!r}."
        )

    return method, layer


def bootstrap_mean_ci(
    values: np.ndarray,
) -> dict[str, float]:
    """
    Estimate a deterministic percentile bootstrap CI for the mean.

    This uncertainty estimate is descriptive only and is never
    used for method or layer selection.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise RuntimeError(
            "Bootstrap received invalid metric values."
        )

    rng = np.random.default_rng(
        SEED
    )

    bootstrap_indices = rng.integers(
        low=0,
        high=len(values),
        size=(
            BOOTSTRAP_REPETITIONS,
            len(values),
        ),
    )

    bootstrap_means = values[
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
        "mean": float(
            values.mean()
        ),
        "bootstrap_95_ci_lower": float(
            lower
        ),
        "bootstrap_95_ci_upper": float(
            upper
        ),
    }


def run_locked_test_evaluation() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    method, layer = load_frozen_xai_selection()

    dataframe = build_source_dataframe()

    test_df = reproduce_locked_test_split(
        dataframe
    )

    models = load_gradcam_models(
        layer
    )

    records: list[dict[str, object]] = []

    print()
    print("=" * 88)
    print("LOCKED TEST XAI CONFIRMATORY EVALUATION")
    print("=" * 88)

    print(
        "Frozen method:",
        method,
    )

    print(
        "Frozen layer:",
        layer,
    )

    print(
        "Locked Test defective samples:",
        len(test_df),
    )

    print(
        "Test used for tuning/selection:",
        False,
    )

    print()
    print(
        f"Evaluating {method} @ {layer} "
        f"{models.conv_model.output_shape}"
    )

    for _, row in test_df.iterrows():
        image_path = Path(
            row["filepath"]
        )

        mask_path = resolve_mask_path(
            source_class=row["source_class"],
            filename=row["filename"],
        )

        with Image.open(
            image_path
        ) as source_image:
            image = source_image.convert(
                "RGB"
            )

        mask = load_binary_mask(
            mask_path
        )

        image_batch = prepare_mobilenet_input(
            image
        )

        # These samples are known defective examples with pixel-level
        # ground-truth masks. Class 0 is targeted to measure defect
        # localization independently of classification correctness.
        heatmap = _make_gradcam_heatmap(
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

        record = {
            "method": method,
            "layer": layer,
            "heatmap_height": int(
                heatmap.shape[0]
            ),
            "heatmap_width": int(
                heatmap.shape[1]
            ),
            "source_partition": row[
                "source_partition"
            ],
            "source_class": row[
                "source_class"
            ],
            "filename": row[
                "filename"
            ],
            **metrics,
        }

        records.append(
            record
        )

        print(
            f"  {row['source_class']:14s} "
            f"{row['filename']:8s} "
            f"AP={metrics['pixel_ap']:.4f} "
            f"SoftIoU={metrics['soft_iou']:.4f} "
            f"Point={int(metrics['pointing_hit'])}"
        )

    results_df = pd.DataFrame(
        records
    )

    if len(results_df) != len(test_df):
        raise RuntimeError(
            "Locked Test evaluation row count mismatch."
        )

    summary_df = pd.DataFrame(
        [
            {
                "method": method,
                "layer": layer,
                "sample_count": len(
                    results_df
                ),
                "mean_pixel_ap": float(
                    results_df[
                        "pixel_ap"
                    ].mean()
                ),
                "std_pixel_ap": float(
                    results_df[
                        "pixel_ap"
                    ].std()
                ),
                "mean_pixel_roc_auc": float(
                    results_df[
                        "pixel_roc_auc"
                    ].mean()
                ),
                "mean_soft_iou": float(
                    results_df[
                        "soft_iou"
                    ].mean()
                ),
                "mean_energy_inside_mask": float(
                    results_df[
                        "energy_inside_mask"
                    ].mean()
                ),
                "mean_energy_gain": float(
                    results_df[
                        "energy_gain"
                    ].mean()
                ),
                "pointing_accuracy": float(
                    results_df[
                        "pointing_hit"
                    ].mean()
                ),
            }
        ]
    )

    class_summary_df = (
        results_df.groupby(
            "source_class",
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
        .sort_values(
            "source_class"
        )
        .reset_index(
            drop=True
        )
    )

    pixel_ap_ci = bootstrap_mean_ci(
        results_df[
            "pixel_ap"
        ].to_numpy()
    )

    soft_iou_ci = bootstrap_mean_ci(
        results_df[
            "soft_iou"
        ].to_numpy()
    )

    results_df.to_csv(
        TEST_PER_IMAGE_PATH,
        index=False,
    )

    summary_df.to_csv(
        TEST_SUMMARY_PATH,
        index=False,
    )

    class_summary_df.to_csv(
        TEST_CLASS_SUMMARY_PATH,
        index=False,
    )

    report = {
        "protocol": (
            "Locked-Test confirmatory XAI localization evaluation"
        ),
        "dataset": "MVTec AD bottle",
        "seed": SEED,
        "frozen_method": method,
        "frozen_layer": layer,
        "heatmap_resolution": [
            int(
                models.conv_model.output_shape[1]
            ),
            int(
                models.conv_model.output_shape[2]
            ),
        ],
        "test_defective_images": int(
            len(results_df)
        ),
        "defect_classes": list(
            DEFECT_CLASSES
        ),
        "selection_source": (
            METHOD_SELECTION_PATH.name
        ),
        "test_used_for_method_selection": False,
        "test_used_for_layer_selection": False,
        "post_test_tuning_permitted": False,
        "metrics": {
            "mean_pixel_ap": float(
                summary_df.iloc[0][
                    "mean_pixel_ap"
                ]
            ),
            "mean_pixel_roc_auc": float(
                summary_df.iloc[0][
                    "mean_pixel_roc_auc"
                ]
            ),
            "mean_soft_iou": float(
                summary_df.iloc[0][
                    "mean_soft_iou"
                ]
            ),
            "mean_energy_inside_mask": float(
                summary_df.iloc[0][
                    "mean_energy_inside_mask"
                ]
            ),
            "mean_energy_gain": float(
                summary_df.iloc[0][
                    "mean_energy_gain"
                ]
            ),
            "pointing_accuracy": float(
                summary_df.iloc[0][
                    "pointing_accuracy"
                ]
            ),
        },
        "uncertainty": {
            "pixel_ap": pixel_ap_ci,
            "soft_iou": soft_iou_ci,
            "bootstrap_repetitions": (
                BOOTSTRAP_REPETITIONS
            ),
        },
        "interpretation_constraint": (
            "Grad-CAM is post-hoc diagnostic evidence and does not "
            "constitute pixel-accurate segmentation or causal proof."
        ),
    }

    TEST_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("LOCKED TEST LOCALIZATION SUMMARY")
    print("=" * 88)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 88)
    print("PER-DEFECT-CLASS SUMMARY")
    print("=" * 88)

    print(
        class_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Pixel AP 95% bootstrap CI: "
        f"[{pixel_ap_ci['bootstrap_95_ci_lower']:.6f}, "
        f"{pixel_ap_ci['bootstrap_95_ci_upper']:.6f}]"
    )

    print(
        "Soft-IoU 95% bootstrap CI: "
        f"[{soft_iou_ci['bootstrap_95_ci_lower']:.6f}, "
        f"{soft_iou_ci['bootstrap_95_ci_upper']:.6f}]"
    )

    print()
    print(
        "XAI CONFIGURATION REMAINS FROZEN:",
        f"{method} @ {layer}",
    )

    print(
        "POST-TEST TUNING:",
        "PROHIBITED",
    )

    print()
    print(
        "Saved:",
        TEST_PER_IMAGE_PATH,
    )

    print(
        "Saved:",
        TEST_SUMMARY_PATH,
    )

    print(
        "Saved:",
        TEST_CLASS_SUMMARY_PATH,
    )

    print(
        "Saved:",
        TEST_REPORT_PATH,
    )


if __name__ == "__main__":
    run_locked_test_evaluation()