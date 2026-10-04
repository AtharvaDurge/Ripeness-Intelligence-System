```markdown
# 🍎 Ripeness Intelligence System

A traditional image-processing based system for **fruit identification and ripeness classification** using handcrafted visual features and Euclidean distance-based classification.

The system analyzes images of:

- 🍎 Apple
- 🍌 Banana
- 🍊 Orange

and classifies them as:

- Ripe
- Unripe

The project uses **OpenCV, HSV color analysis, histograms, GLCM texture features, shape features, feature normalization, and reference-vector comparison**.

> **This project does not use Machine Learning or Deep Learning classifiers.**

---

## 🎯 Objective

The objective of this project is to determine the **fruit type and ripeness condition** of a fruit image using traditional digital image processing techniques.

Instead of training a conventional classifier, the system creates reference profiles for each fruit-ripeness class and compares a new image against those profiles using **Euclidean distance**.

---

## 🧠 System Overview

```text
                Input Fruit Image
                        │
                        ▼
                 Image Resizing
                        │
                        ▼
                 Gaussian Blur
                        │
                        ▼
                 HSV Conversion
                        │
                        ▼
                Fruit Segmentation
                        │
                        ▼
              Feature Extraction
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        Color       Histogram      Texture
       Features      Features       Features
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                  Shape Features
                        │
                        ▼
                64-Dimensional
                 Feature Vector
                        │
                        ▼
               Feature Normalization
                        │
                        ▼
              Euclidean Distance
                        │
                        ▼
             Reference Profiles
                        │
                        ▼
             Closest Class Selected
                        │
                        ▼
             Fruit + Ripeness Result
```

---

# 🔬 Feature Extraction

Each input image is converted into a **64-dimensional feature vector**.

## 1. Color Features — 6

The system extracts statistical information from the HSV color representation.

Features:

- Mean Hue
- Standard Deviation of Hue
- Mean Saturation
- Standard Deviation of Saturation
- Mean Value
- Standard Deviation of Value

```text
Color Features = 6
```

---

## 2. HSV Histogram Features — 48

Normalized histograms are calculated for:

- Hue
- Saturation
- Value

Each channel uses 16 bins.

```text
Hue Histogram        = 16
Saturation Histogram = 16
Value Histogram      = 16
--------------------------------
Total                = 48
```

The histograms represent the **distribution of color values within the fruit**.

---

## 3. Texture Features — 4

Texture information is extracted using the **Gray-Level Co-occurrence Matrix (GLCM)**.

The four extracted properties are:

- Contrast
- Correlation
- Energy
- Homogeneity

```text
Texture Features = 4
```

---

## 4. Shape Features — 6

The segmented fruit contour is used to calculate:

- Area Ratio
- Normalized Perimeter
- Circularity
- Aspect Ratio
- Solidity
- Extent

```text
Shape Features = 6
```

---

## 📊 Total Feature Vector

```text
6 Color Features
        +
48 Histogram Features
        +
4 Texture Features
        +
6 Shape Features
        =
64 Features
```

Every image is therefore represented as:

```text
[ f1, f2, f3, ... , f64 ]
```

---

# 📐 Classification Method

The system uses a **reference-vector classification approach**.

It does not use:

- Random Forest
- SVM
- KNN
- Neural Networks
- CNN
- Deep Learning

Instead, reference vectors are created for six classes:

```text
Apple  - Ripe
Apple  - Unripe

Banana - Ripe
Banana - Unripe

Orange - Ripe
Orange - Unripe
```

---

## 🏋️ Reference Vector Creation

During the training stage:

### Step 1 — Feature Extraction

64 features are extracted from every training image.

### Step 2 — Calculate Statistics

The global mean and standard deviation of every feature are calculated.

### Step 3 — Feature Normalization

Each feature is normalized using:

```text
z = (x - μ) / σ
```

where:

```text
x = original feature value
μ = feature mean
σ = feature standard deviation
```

### Step 4 — Create Class Prototypes

The normalized feature vectors belonging to each class are averaged.

This produces one reference vector for every fruit-ripeness class.

---

# 📏 Euclidean Distance

For a new input image, its normalized feature vector is compared with every reference vector.

The Euclidean distance is:

```text
              __________________
             /
d =          √ Σ (xi - yi)²
```

where:

```text
x = input image feature vector
y = reference feature vector
d = Euclidean distance
```

The class with the **smallest distance** is selected as the prediction.

```text
Smaller Distance
       ↓
Greater Similarity
       ↓
Selected Class
```

---

# 🖥️ Streamlit Application

The project includes a Streamlit-based graphical interface.

The application provides:

### Prediction

Displays:

- Fruit
- Ripeness
- Relative similarity score

### Reference Distance

Displays:

- Closest reference
- Distance to closest reference
- Distance to second-closest reference

### Image Processing Pipeline

Displays:

```text
Original Image
       ↓
Gaussian Blurred Image
       ↓
Segmentation Mask
```

### Feature Analysis

Displays all **64 extracted features**.

### HSV Histograms

Displays:

- Hue Histogram
- Saturation Histogram
- Value / Brightness Histogram

### Reference Comparison

Displays the Euclidean distance from the uploaded image to all six reference profiles.

---

# 📁 Project Structure

```text
Ripeness-Intelligence-System/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/              # Not included in repository
│   └── dataset1/
│       ├── train/
│       └── test/
│
└── artifacts/            # Generated locally, not included
    └── model.npz
```

### Dataset

The dataset is intentionally **not included** in this repository.

### Artifacts

The generated reference model:

```text
artifacts/model.npz
```

is also intentionally excluded.

Both directories are listed in `.gitignore`.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AtharvaDurge/Ripeness-Intelligence-System.git
```

Navigate to the project:

```bash
cd Ripeness-Intelligence-System
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset Setup

The dataset is not included in this repository.

Place the dataset inside:

```text
dataset/
```

The expected structure is:

```text
dataset/
└── dataset1/
    │
    ├── train/
    │   │
    │   ├── apples/
    │   │   ├── ripe/
    │   │   └── raw/
    │   │
    │   ├── banana/
    │   │   ├── ripe/
    │   │   └── unripe/
    │   │
    │   └── orange/
    │       ├── ripe/
    │       └── unripe/
    │
    └── test/
        │
        ├── apple/
        │   ├── ripe/
        │   └── unripe/
        │
        ├── banana/
        │   ├── ripe/
        │   └── unripe/
        │
        └── orange/
            ├── ripe/
            └── unripe/
```

---

# 🏋️ Generate the Reference Model

After placing the dataset in the correct location:

```bash
python main.py train
```

The program will:

1. Load the training images.
2. Extract 64 features from each image.
3. Calculate feature statistics.
4. Normalize the feature vectors.
5. Create six reference vectors.
6. Save the reference model.

The generated model will be saved as:

```text
artifacts/model.npz
```

---

# 📊 Evaluate the System

To evaluate the system using the test dataset:

```bash
python main.py evaluate
```

The evaluation provides:

- Overall classification accuracy
- Fruit-wise accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix

---

# 🌐 Run the Application

After generating the reference model:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

Upload an image of an:

```text
Apple
Banana
Orange
```

and the system will perform the complete image-processing pipeline and return the predicted fruit and ripeness.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| Pandas | Data handling |
| Matplotlib | Histogram visualization |
| scikit-image | GLCM texture analysis |
| Streamlit | Web interface |

---

# 📚 Image Processing Techniques

The project demonstrates:

- Image resizing
- Gaussian filtering
- HSV color-space conversion
- LAB color-space conversion
- Color statistics
- Histogram analysis
- Otsu thresholding
- Morphological opening
- Morphological closing
- Contour detection
- GLCM texture analysis
- Shape analysis
- Feature normalization
- Euclidean distance classification

---

# 🔄 Complete Workflow

```text
Training Images
      │
      ▼
Preprocessing
      │
      ▼
Segmentation
      │
      ▼
64 Feature Extraction
      │
      ▼
Normalization
      │
      ▼
Six Reference Vectors
      │
      │
      ▼
────────────────────────
      │
      │  New Image
      ▼
Preprocessing
      │
      ▼
Segmentation
      │
      ▼
64 Feature Extraction
      │
      ▼
Normalization
      │
      ▼
Euclidean Distance
      │
      ▼
Closest Reference
      │
      ▼
Fruit + Ripeness
```

---

# 🎯 Project Highlights

- **Traditional image processing approach**
- **No conventional ML classifier**
- **64 handcrafted features**
- **Color + histogram + texture + shape analysis**
- **Six reference profiles**
- **Euclidean distance-based classification**
- **Interactive Streamlit interface**
- **Visual feature analysis**
- **Test-set evaluation**

---

# 🔮 Future Improvements

Possible extensions include:

- Support for additional fruit categories
- Improved fruit segmentation
- Larger and more diverse datasets
- Additional handcrafted features
- Alternative distance metrics
- Real-time camera input
- Mobile or web deployment
- Comparison with other traditional classifiers

---

# 👨‍💻 Author

**Atharva Durge**

Engineering Student  
VJTI, Mumbai

---

