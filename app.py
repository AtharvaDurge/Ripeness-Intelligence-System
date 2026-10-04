import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from main import (
    load_model,
    predict_from_features,
    extract_features_from_bgr,
    FEATURE_NAMES
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fruit Ripeness Analyzer",
    page_icon="🍎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 18px;
    }

    .result-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🍎 Fruit Ripeness Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Traditional Image Processing Based Fruit Ripeness Classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This system uses traditional image
        processing rather than machine learning.
        """
    )

    st.write(
        """
        ### Processing

        1. Resize
        2. Gaussian Blur
        3. HSV Conversion
        4. Fruit Segmentation
        5. Color Features
        6. HSV Histograms
        7. GLCM Texture
        8. Shape Features
        9. Feature Normalization
        10. Euclidean Distance
        """
    )

    st.write(
        """
        ### Features

        **64 features per image**

        • 6 color statistics

        • 48 histogram features

        • 4 texture features

        • 6 shape features
        """
    )


# ============================================================
# LOAD REFERENCE MODEL
# ============================================================

try:

    model = load_model()

except FileNotFoundError:

    st.error(
        """
        ⚠️ Reference model not found.

        First run:

        `python main.py train`

        Then restart the Streamlit application.
        """
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("Upload Fruit Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload an Apple, Banana, or Orange image "
        "to begin analysis."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

file_bytes = uploaded_file.read()

image_array = np.frombuffer(
    file_bytes,
    dtype=np.uint8
)

image = cv2.imdecode(
    image_array,
    cv2.IMREAD_COLOR
)


if image is None:

    st.error(
        "Could not read the uploaded image."
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

with st.spinner("Processing image..."):

    features, debug = extract_features_from_bgr(
        image
    )

    result = predict_from_features(
        features,
        model
    )


# ============================================================
# PREDICTION
# ============================================================

st.header("Prediction")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Fruit",
        result["fruit"].title()
    )


with col2:

    st.metric(
        "Ripeness",
        result["ripeness"].title()
    )


with col3:

    st.metric(
        "Relative Score",
        f"{result['relative_score'] * 100:.1f}%"
    )


st.caption(
    "The score shown is a relative similarity score "
    "based on prototype distances, not a calibrated probability."
)


# ============================================================
# REFERENCE DISTANCE
# ============================================================

st.subheader("Reference Distance")

st.write(
    f"Closest reference: "
    f"**{result['fruit'].title()} - "
    f"{result['ripeness'].title()}**"
)

st.write(
    f"Distance to closest reference: "
    f"**{result['distance']:.4f}**"
)

st.write(
    f"Distance to second closest reference: "
    f"**{result['second_distance']:.4f}**"
)


# ============================================================
# IMAGE PROCESSING PIPELINE
# ============================================================

st.header("Image Processing Pipeline")

original_rgb = cv2.cvtColor(
    debug["original"],
    cv2.COLOR_BGR2RGB
)

blurred_rgb = cv2.cvtColor(
    debug["blurred"],
    cv2.COLOR_BGR2RGB
)

mask = debug["mask"]


col1, col2, col3 = st.columns(3)


with col1:

    st.image(
        original_rgb,
        caption="1. Original Image",
        use_container_width=True
    )


with col2:

    st.image(
        blurred_rgb,
        caption="2. Gaussian Blurred",
        use_container_width=True
    )


with col3:

    st.image(
        mask,
        caption="3. Segmentation Mask",
        use_container_width=True
    )


# ============================================================
# DISTANCE FROM ALL REFERENCE PROFILES
# ============================================================

st.header(
    "Distance from All Reference Profiles"
)

distance_df = pd.DataFrame(
    list(
        result["distances"].items()
    ),
    columns=[
        "Reference Class",
        "Euclidean Distance"
    ]
)

distance_df = (
    distance_df
    .sort_values("Euclidean Distance")
    .reset_index(drop=True)
)


st.dataframe(
    distance_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE TABLE
# ============================================================

st.header("Extracted 64 Features")

feature_df = pd.DataFrame(
    {
        "Feature": FEATURE_NAMES,
        "Value": features
    }
)

st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True,
    height=450
)


# ============================================================
# FEATURE GROUPS
# ============================================================

st.header("Feature Groups")

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Color Features",
        "6"
    )


with c2:

    st.metric(
        "Histogram Features",
        "48"
    )


with c3:

    st.metric(
        "Texture Features",
        "4"
    )


with c4:

    st.metric(
        "Shape Features",
        "6"
    )


st.success(
    "Total feature vector size = 64"
)


# ============================================================
# HSV HISTOGRAMS
# ============================================================

st.header("HSV Histograms")

hue_hist = features[6:22]

saturation_hist = features[22:38]

value_hist = features[38:54]


# ------------------------------------------------------------
# HUE HISTOGRAM
# ------------------------------------------------------------

st.subheader("Hue Distribution")

fig, ax = plt.subplots(
    figsize=(4.8, 2.8),
    dpi=100
)

ax.bar(
    range(1, 17),
    hue_hist
)

ax.set_xlabel(
    "Hue Bin",
    fontsize=9
)

ax.set_ylabel(
    "Normalized Frequency",
    fontsize=9
)

ax.set_title(
    "Hue Histogram",
    fontsize=11
)

ax.grid(
    axis="y",
    alpha=0.25
)

st.pyplot(
    fig,
    clear_figure=True,
    use_container_width=False
)


# ------------------------------------------------------------
# SATURATION HISTOGRAM
# ------------------------------------------------------------

st.subheader(
    "Saturation Distribution"
)

fig, ax = plt.subplots(
    figsize=(4.8, 2.8),
    dpi=100
)

ax.bar(
    range(1, 17),
    saturation_hist
)

ax.set_xlabel(
    "Saturation Bin",
    fontsize=9
)

ax.set_ylabel(
    "Normalized Frequency",
    fontsize=9
)

ax.set_title(
    "Saturation Histogram",
    fontsize=11
)

ax.grid(
    axis="y",
    alpha=0.25
)

st.pyplot(
    fig,
    clear_figure=True,
    use_container_width=False
)


# ------------------------------------------------------------
# VALUE HISTOGRAM
# ------------------------------------------------------------

st.subheader(
    "Brightness / Value Distribution"
)

fig, ax = plt.subplots(
    figsize=(4.8, 2.8),
    dpi=100
)

ax.bar(
    range(1, 17),
    value_hist
)

ax.set_xlabel(
    "Value Bin",
    fontsize=9
)

ax.set_ylabel(
    "Normalized Frequency",
    fontsize=9
)

ax.set_title(
    "Value Histogram",
    fontsize=11
)

ax.grid(
    axis="y",
    alpha=0.25
)

st.pyplot(
    fig,
    clear_figure=True,
    use_container_width=False
)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander("View Technical Details"):

    st.write(
        "### Classification Method"
    )

    st.write(
        """
        Each image is represented by a
        64-dimensional feature vector.

        The vector is normalized using the mean
        and standard deviation calculated from
        the training dataset.

        Euclidean distance is then calculated
        between the test image and six reference
        vectors:

        • Apple - Ripe

        • Apple - Unripe

        • Banana - Ripe

        • Banana - Unripe

        • Orange - Ripe

        • Orange - Unripe

        The class with the smallest distance
        is selected.
        """
    )

    st.write(
        "### Feature Vector"
    )

    st.code(
        "6 Color + 48 Histogram + "
        "4 Texture + 6 Shape = 64 Features"
    )