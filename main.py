import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

# Automatically detect all datasets inside the dataset folder
DATASET_ROOTS = sorted(
    [p for p in DATASET_DIR.iterdir() if p.is_dir()]
) if DATASET_DIR.exists() else []

MODEL_DIR = BASE_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "model.npz"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (256, 256)

FRUITS = ["apple", "banana", "orange"]

CLASSES = ["ripe", "unripe"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "Mean_Hue",
    "Std_Hue",
    "Mean_Saturation",
    "Std_Saturation",
    "Mean_Value",
    "Std_Value"
]

FEATURE_NAMES += [
    f"Hue_Hist_{i + 1}"
    for i in range(16)
]

FEATURE_NAMES += [
    f"Saturation_Hist_{i + 1}"
    for i in range(16)
]

FEATURE_NAMES += [
    f"Value_Hist_{i + 1}"
    for i in range(16)
]

FEATURE_NAMES += [
    "GLCM_Contrast",
    "GLCM_Correlation",
    "GLCM_Energy",
    "GLCM_Homogeneity"
]

FEATURE_NAMES += [
    "Area_Ratio",
    "Normalized_Perimeter",
    "Circularity",
    "Aspect_Ratio",
    "Solidity",
    "Extent"
]

assert len(FEATURE_NAMES) == 64, (
    f"Expected 64 features, got {len(FEATURE_NAMES)}"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load image using OpenCV.

    Returns:
        BGR image or None if loading fails.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read image: {image_path}")

    return image


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Preprocess image.

    Steps:
        1. Resize
        2. Gaussian Blur
        3. Convert BGR -> HSV
        4. Convert BGR -> Grayscale
    """

    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    blurred = cv2.GaussianBlur(
        resized,
        (5, 5),
        0
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    return resized, blurred, hsv, gray


# ============================================================
# FRUIT SEGMENTATION
# ============================================================

def segment_fruit(image):
    """
    Segment fruit from background.

    Approach:
        1. Convert image to LAB
        2. Estimate background color using image border
        3. Calculate color distance
        4. Otsu threshold
        5. Morphological opening
        6. Morphological closing
        7. Keep largest connected contour
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    h, w = lab.shape[:2]

    border_size = 10

    top = lab[:border_size, :, :]
    bottom = lab[h - border_size:h, :, :]
    left = lab[:, :border_size, :]
    right = lab[:, w - border_size:w, :]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ],
        axis=0
    )

    background_color = np.median(
        border_pixels,
        axis=0
    )

    diff = (
        lab.astype(np.float32)
        - background_color
    )

    distance = np.sqrt(
        np.sum(diff ** 2, axis=2)
    )

    distance_normalized = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, mask = cv2.threshold(
        distance_normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(mask)

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    final_mask = np.zeros_like(mask)

    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=-1
    )

    return final_mask


# ============================================================
# COLOR FEATURES
# ============================================================

def extract_color_features(hsv, mask):
    """
    Extract:

        Mean Hue
        Std Hue
        Mean Saturation
        Std Saturation
        Mean Value
        Std Value

    TOTAL = 6
    """

    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        pixels = hsv.reshape(-1, 3)

    hue = pixels[:, 0].astype(np.float32)
    saturation = pixels[:, 1].astype(np.float32)
    value = pixels[:, 2].astype(np.float32)

    features = [
        np.mean(hue),
        np.std(hue),
        np.mean(saturation),
        np.std(saturation),
        np.mean(value),
        np.std(value)
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# HISTOGRAM FEATURES
# ============================================================

def normalized_histogram(values, bins, value_range):
    """
    Calculate normalized histogram.
    """

    hist, _ = np.histogram(
        values,
        bins=bins,
        range=value_range
    )

    hist = hist.astype(np.float32)

    total = np.sum(hist)

    if total > 0:
        hist /= total

    return hist


def extract_histogram_features(hsv, mask):
    """
    Extract:

        Hue histogram        = 16
        Saturation histogram = 16
        Value histogram      = 16

    TOTAL = 48
    """

    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        pixels = hsv.reshape(-1, 3)

    hue = pixels[:, 0]
    saturation = pixels[:, 1]
    value = pixels[:, 2]

    hue_hist = normalized_histogram(
        hue,
        16,
        (0, 180)
    )

    saturation_hist = normalized_histogram(
        saturation,
        16,
        (0, 256)
    )

    value_hist = normalized_histogram(
        value,
        16,
        (0, 256)
    )

    return np.concatenate(
        [
            hue_hist,
            saturation_hist,
            value_hist
        ]
    ).astype(np.float32)


# ============================================================
# TEXTURE FEATURES
# ============================================================

def extract_texture_features(gray, mask):
    """
    Extract GLCM texture features:

        Contrast
        Correlation
        Energy
        Homogeneity

    TOTAL = 4
    """

    try:
        from skimage.feature import (
            graycomatrix,
            graycoprops
        )

    except ImportError:
        raise ImportError(
            "scikit-image is required for GLCM features.\n"
            "Install it using:\n"
            "pip install scikit-image"
        )

    gray_copy = gray.copy()

    fruit_pixels = gray_copy[mask > 0]

    if len(fruit_pixels) == 0:
        fruit_median = np.median(gray_copy)

    else:
        fruit_median = np.median(fruit_pixels)

    gray_copy[mask == 0] = fruit_median

    gray_quantized = (
        gray_copy.astype(np.float32)
        / 256.0
        * 32
    ).astype(np.uint8)

    gray_quantized = np.clip(
        gray_quantized,
        0,
        31
    )

    distances = [1]

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        gray_quantized,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = np.mean(
        graycoprops(
            glcm,
            "contrast"
        )
    )

    correlation = np.mean(
        graycoprops(
            glcm,
            "correlation"
        )
    )

    energy = np.mean(
        graycoprops(
            glcm,
            "energy"
        )
    )

    homogeneity = np.mean(
        graycoprops(
            glcm,
            "homogeneity"
        )
    )

    return np.array(
        [
            contrast,
            correlation,
            energy,
            homogeneity
        ],
        dtype=np.float32
    )


# ============================================================
# SHAPE FEATURES
# ============================================================

def extract_shape_features(mask):
    """
    Extract:

        Area Ratio
        Normalized Perimeter
        Circularity
        Aspect Ratio
        Solidity
        Extent

    TOTAL = 6
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(
            6,
            dtype=np.float32
        )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    image_area = (
        mask.shape[0]
        * mask.shape[1]
    )

    area_ratio = (
        area / image_area
        if image_area > 0
        else 0
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    image_diagonal = np.sqrt(
        mask.shape[0] ** 2
        + mask.shape[1] ** 2
    )

    normalized_perimeter = (
        perimeter / image_diagonal
        if image_diagonal > 0
        else 0
    )

    if perimeter > 0:
        circularity = (
            4
            * np.pi
            * area
            / (perimeter ** 2)
        )

    else:
        circularity = 0

    x, y, width, height = cv2.boundingRect(
        contour
    )

    if height > 0:
        aspect_ratio = width / height

    else:
        aspect_ratio = 0

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = area / hull_area

    else:
        solidity = 0

    rectangle_area = width * height

    if rectangle_area > 0:
        extent = area / rectangle_area

    else:
        extent = 0

    return np.array(
        [
            area_ratio,
            normalized_perimeter,
            circularity,
            aspect_ratio,
            solidity,
            extent
        ],
        dtype=np.float32
    )


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_features_from_bgr(image):
    """
    Complete image processing pipeline.

    Returns:
        feature_vector = 64 features

        debug dictionary containing:
            original
            blurred
            hsv
            gray
            mask
    """

    resized, blurred, hsv, gray = preprocess_image(
        image
    )

    mask = segment_fruit(
        blurred
    )

    color_features = extract_color_features(
        hsv,
        mask
    )

    histogram_features = extract_histogram_features(
        hsv,
        mask
    )

    texture_features = extract_texture_features(
        gray,
        mask
    )

    shape_features = extract_shape_features(
        mask
    )

    features = np.concatenate(
        [
            color_features,
            histogram_features,
            texture_features,
            shape_features
        ]
    ).astype(np.float32)

    if len(features) != 64:
        raise RuntimeError(
            f"Feature extraction returned "
            f"{len(features)} features instead of 64."
        )

    debug = {
        "original": resized,
        "blurred": blurred,
        "hsv": hsv,
        "gray": gray,
        "mask": mask
    }

    return features, debug


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_samples(split="train"):
    """
    Collect image paths from all datasets.

    Apple training structure:

        train/apples/ripe
        train/apples/raw

    Apple testing structure:

        test/apple/ripe
        test/apple/unripe

    Banana and orange use:

        train/banana/ripe
        train/banana/unripe

        train/orange/ripe
        train/orange/unripe
    """

    samples = []

    for dataset_root in DATASET_ROOTS:

        split_dir = (
            dataset_root
            / split
        )

        if not split_dir.exists():
            print(
                f"Warning: folder not found: "
                f"{split_dir}"
            )
            continue

        # ----------------------------------------------------
        # Apple special training structure
        # ----------------------------------------------------

        apples_dir = split_dir / "apples"

        if apples_dir.exists():

            ripe_dir = apples_dir / "ripe"

            if ripe_dir.exists():

                for image_path in sorted(
                    ripe_dir.iterdir()
                ):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in VALID_EXTENSIONS
                    ):

                        samples.append(
                            {
                                "path": image_path,
                                "fruit": "apple",
                                "ripeness": "ripe",
                                "class_name": "apple_ripe"
                            }
                        )

            raw_dir = apples_dir / "raw"

            if raw_dir.exists():

                for image_path in sorted(
                    raw_dir.iterdir()
                ):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in VALID_EXTENSIONS
                    ):

                        samples.append(
                            {
                                "path": image_path,
                                "fruit": "apple",
                                "ripeness": "unripe",
                                "class_name": "apple_unripe"
                            }
                        )

        # ----------------------------------------------------
        # Apple normal structure
        # ----------------------------------------------------

        apple_dir = split_dir / "apple"

        if apple_dir.exists():

            for ripeness in [
                "ripe",
                "unripe"
            ]:

                class_dir = (
                    apple_dir
                    / ripeness
                )

                if not class_dir.exists():
                    continue

                for image_path in sorted(
                    class_dir.iterdir()
                ):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in VALID_EXTENSIONS
                    ):

                        samples.append(
                            {
                                "path": image_path,
                                "fruit": "apple",
                                "ripeness": ripeness,
                                "class_name": f"apple_{ripeness}"
                            }
                        )

        # ----------------------------------------------------
        # Banana
        # ----------------------------------------------------

        banana_dir = split_dir / "banana"

        if banana_dir.exists():

            for ripeness in [
                "ripe",
                "unripe"
            ]:

                class_dir = (
                    banana_dir
                    / ripeness
                )

                if not class_dir.exists():
                    continue

                for image_path in sorted(
                    class_dir.iterdir()
                ):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in VALID_EXTENSIONS
                    ):

                        samples.append(
                            {
                                "path": image_path,
                                "fruit": "banana",
                                "ripeness": ripeness,
                                "class_name": f"banana_{ripeness}"
                            }
                        )

        # ----------------------------------------------------
        # Orange
        # ----------------------------------------------------

        orange_dir = split_dir / "orange"

        if orange_dir.exists():

            for ripeness in [
                "ripe",
                "unripe"
            ]:

                class_dir = (
                    orange_dir
                    / ripeness
                )

                if not class_dir.exists():
                    continue

                for image_path in sorted(
                    class_dir.iterdir()
                ):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in VALID_EXTENSIONS
                    ):

                        samples.append(
                            {
                                "path": image_path,
                                "fruit": "orange",
                                "ripeness": ripeness,
                                "class_name": f"orange_{ripeness}"
                            }
                        )

    return samples


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

def print_dataset_distribution(samples):
    """
    Print number of images in every class.
    """

    print()
    print("Dataset distribution:")
    print("-" * 60)

    for fruit in FRUITS:

        for ripeness in CLASSES:

            class_name = (
                f"{fruit}_{ripeness}"
            )

            count = sum(
                1
                for sample in samples
                if sample["class_name"]
                == class_name
            )

            print(
                f"{class_name:<20} : "
                f"{count:>5} images"
            )

    print("-" * 60)

    print(
        f"{'TOTAL':<20} : "
        f"{len(samples):>5} images"
    )

    print()


# ============================================================
# TRAIN / CREATE REFERENCE VECTORS
# ============================================================

def train_model():
    """
    Create reference vectors for each
    fruit-ripeness class.

    This is NOT conventional machine learning.

    We:

        1. Extract 64 features.
        2. Calculate global mean and standard deviation.
        3. Normalize features.
        4. Average normalized vectors within each class.
        5. Store the average as the reference vector.
    """

    print("=" * 65)
    print("TRAINING / REFERENCE CREATION")
    print("=" * 65)

    print("\nDataset roots:")

    for root in DATASET_ROOTS:
        print(f"  {root}")

    samples = collect_samples(
        split="train"
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No training images found."
        )

    print(
        f"\nTraining images: "
        f"{len(samples)}"
    )

    print_dataset_distribution(
        samples
    )

    all_features = []

    processed_samples = []

    total = len(samples)

    for index, sample in enumerate(
        samples,
        start=1
    ):

        image = load_image(
            sample["path"]
        )

        if image is None:
            continue

        try:

            features, _ = extract_features_from_bgr(
                image
            )

        except Exception as e:

            print(
                f"\nWARNING: Feature extraction "
                f"failed for:\n"
                f"{sample['path']}\n"
                f"Error: {e}"
            )

            continue

        all_features.append(
            features
        )

        processed_samples.append(
            sample
        )

        if (
            index % 25 == 0
            or index == total
        ):

            print(
                f"Processed "
                f"{index}/{total}"
            )

    if len(all_features) == 0:
        raise RuntimeError(
            "No images were successfully processed."
        )

    X = np.vstack(
        all_features
    ).astype(np.float32)

    feature_mean = np.mean(
        X,
        axis=0
    )

    feature_std = np.std(
        X,
        axis=0
    )

    feature_std[
        feature_std < 1e-8
    ] = 1.0

    X_normalized = (
        X - feature_mean
    ) / feature_std

    print()
    print(
        "Creating reference vectors..."
    )

    class_names = []

    prototypes = []

    for fruit in FRUITS:

        for ripeness in CLASSES:

            class_name = (
                f"{fruit}_{ripeness}"
            )

            class_indices = [
                i
                for i, sample
                in enumerate(
                    processed_samples
                )
                if sample["class_name"]
                == class_name
            ]

            if len(class_indices) == 0:
                raise RuntimeError(
                    f"No training images found "
                    f"for {class_name}"
                )

            class_vectors = (
                X_normalized[
                    class_indices
                ]
            )

            prototype = np.mean(
                class_vectors,
                axis=0
            )

            class_names.append(
                class_name
            )

            prototypes.append(
                prototype
            )

            print(
                f"{class_name:<20} : "
                f"{len(class_indices):>4} images"
            )

    prototypes = np.vstack(
        prototypes
    ).astype(np.float32)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        MODEL_PATH,
        feature_mean=feature_mean,
        feature_std=feature_std,
        prototypes=prototypes,
        class_names=np.array(
            class_names
        ),
        feature_names=np.array(
            FEATURE_NAMES
        )
    )

    print()
    print("=" * 65)
    print("REFERENCE CREATION COMPLETE")
    print("=" * 65)

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        f"Classes: "
        f"{', '.join(class_names)}"
    )

    print()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load saved reference vectors.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model not found.\n\n"
            "Run:\n"
            "python main.py train"
        )

    data = np.load(
        MODEL_PATH,
        allow_pickle=True
    )

    model = {
        "feature_mean": data[
            "feature_mean"
        ],
        "feature_std": data[
            "feature_std"
        ],
        "prototypes": data[
            "prototypes"
        ],
        "class_names": data[
            "class_names"
        ].tolist(),
        "feature_names": data[
            "feature_names"
        ].tolist()
    }

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_from_features(
    features,
    model
):
    """
    Predict class using minimum
    Euclidean distance.
    """

    feature_mean = model[
        "feature_mean"
    ]

    feature_std = model[
        "feature_std"
    ]

    prototypes = model[
        "prototypes"
    ]

    class_names = model[
        "class_names"
    ]

    normalized_features = (
        features - feature_mean
    ) / feature_std

    distances = np.linalg.norm(
        prototypes
        - normalized_features,
        axis=1
    )

    sorted_indices = np.argsort(
        distances
    )

    best_index = int(
        sorted_indices[0]
    )

    second_index = int(
        sorted_indices[1]
    )

    predicted_class = (
        class_names[best_index]
    )

    parts = predicted_class.split(
        "_"
    )

    fruit = parts[0]

    ripeness = parts[1]

    closest_distance = float(
        distances[best_index]
    )

    second_distance = float(
        distances[second_index]
    )

    inverse_distances = (
        1.0
        / (distances + 1e-8)
    )

    relative_scores = (
        inverse_distances
        / np.sum(inverse_distances)
    )

    relative_score = float(
        relative_scores[best_index]
    )

    distance_dict = {
        class_names[i]: float(
            distances[i]
        )
        for i in range(
            len(class_names)
        )
    }

    return {
        "class_name": predicted_class,
        "fruit": fruit,
        "ripeness": ripeness,
        "distance": closest_distance,
        "second_distance": second_distance,
        "distances": distance_dict,
        "relative_score": relative_score
    }


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image):
    """
    Extract features and predict an image.
    """

    model = load_model()

    features, debug = extract_features_from_bgr(
        image
    )

    result = predict_from_features(
        features,
        model
    )

    result["features"] = features
    result["debug"] = debug

    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model():
    """
    Evaluate the reference-vector classifier
    on the test dataset.
    """

    print("=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    model = load_model()

    samples = collect_samples(
        split="test"
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No test images found."
        )

    print(
        f"Test images: "
        f"{len(samples)}"
    )

    print_dataset_distribution(
        samples
    )

    y_true = []
    y_pred = []

    processed = 0

    for sample in samples:

        image = load_image(
            sample["path"]
        )

        if image is None:
            continue

        try:

            features, _ = extract_features_from_bgr(
                image
            )

            result = predict_from_features(
                features,
                model
            )

        except Exception as e:

            print(
                f"WARNING: Could not process "
                f"{sample['path']}: {e}"
            )

            continue

        y_true.append(
            sample["class_name"]
        )

        y_pred.append(
            result["class_name"]
        )

        processed += 1

    print()

    print(
        f"Successfully evaluated: "
        f"{processed}/{len(samples)}"
    )

    if len(y_true) == 0:
        raise RuntimeError(
            "No test images were successfully evaluated."
        )

    correct = sum(
        true == pred
        for true, pred
        in zip(y_true, y_pred)
    )

    overall_accuracy = (
        correct / len(y_true)
    )

    print()
    print("=" * 65)
    print("SIX-CLASS CLASSIFICATION")
    print("=" * 65)

    print(
        f"Accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print("=" * 65)
    print("RIPE vs UNRIPE EVALUATION")
    print("=" * 65)

    for fruit in FRUITS:

        fruit_true = []
        fruit_pred = []

        for true, pred in zip(
            y_true,
            y_pred
        ):

            true_parts = true.split(
                "_"
            )

            pred_parts = pred.split(
                "_"
            )

            if true_parts[0] == fruit:

                fruit_true.append(
                    true_parts[1]
                )

                fruit_pred.append(
                    pred_parts[1]
                    if pred_parts[0] == fruit
                    else "wrong_fruit"
                )

        if len(fruit_true) == 0:
            continue

        tp = sum(
            1
            for t, p
            in zip(
                fruit_true,
                fruit_pred
            )
            if t == "ripe"
            and p == "ripe"
        )

        tn = sum(
            1
            for t, p
            in zip(
                fruit_true,
                fruit_pred
            )
            if t == "unripe"
            and p == "unripe"
        )

        fp = sum(
            1
            for t, p
            in zip(
                fruit_true,
                fruit_pred
            )
            if t == "unripe"
            and p == "ripe"
        )

        fn = sum(
            1
            for t, p
            in zip(
                fruit_true,
                fruit_pred
            )
            if t == "ripe"
            and p != "ripe"
        )

        total = len(
            fruit_true
        )

        accuracy = (
            (tp + tn) / total
            if total > 0
            else 0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print()
        print(
            f"{fruit.upper()}"
        )

        print(
            f"Accuracy  : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score  : "
            f"{f1 * 100:.2f}%"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        print(
            "                 Predicted"
        )

        print(
            "                 Unripe     Ripe"
        )

        print(
            f"Actual Unripe    "
            f"{tn:>5}     {fp:>5}"
        )

        print(
            f"Actual Ripe      "
            f"{fn:>5}     {tp:>5}"
        )

    print()
    print("=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    import sys

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print("  python main.py train")
        print("  python main.py evaluate")
        print()

        return

    command = sys.argv[1].lower()

    if command == "train":

        train_model()

    elif command == "evaluate":

        evaluate_model()

    else:

        print(
            f"Unknown command: "
            f"{command}"
        )

        print()

        print("Use:")
        print("  python main.py train")
        print("  python main.py evaluate")


if __name__ == "__main__":
    main()