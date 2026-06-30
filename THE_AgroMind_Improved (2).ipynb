{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.0"
    },
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k1kSclTqaUUH"
      },
      "source": [
        "# 🌿 AgroMind – Plant Disease Classification Pipeline\n",
        "\n",
        "**Dataset:** PlantVillage (Kaggle)  \n",
        "**Framework:** PyTorch + Albumentations + MLflow  \n",
        "**Authors:** AgroMind Team\n",
        "\n",
        "---\n",
        "### 📋 Notebook Structure\n",
        "| # | Section | Purpose |\n",
        "|---|---------|---------|\n",
        "| 1 | Environment Setup | Imports, seeds, device |\n",
        "| 2 | Dataset Discovery | Scan PlantVillage folder structure |\n",
        "| 3 | Data Quality & Splitting | Blur filter, stratified 70/15/15 split |\n",
        "| 4 | Exploratory Data Analysis | Class distribution, sample images |\n",
        "| 5 | Augmentation Pipeline | Train vs val/test transforms |\n",
        "| 6 | Dataset & DataLoader | PyTorch Dataset, imbalance strategies |\n",
        "| 7 | Model Architecture | BaselineCNN + pretrained backbones |\n",
        "| 8 | Training Loop | Early stopping, LR scheduling, MLflow |\n",
        "| 9 | Evaluation | Classification report, confusion matrix |\n"
      ],
      "id": "k1kSclTqaUUH"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7BFvOVfcaUUJ"
      },
      "source": [
        "---\n",
        "## 1️⃣ Environment Setup\n",
        "Install dependencies, configure seeds, and detect the compute device.\n"
      ],
      "id": "7BFvOVfcaUUJ"
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KyDKkEHyaUUJ",
        "outputId": "5ec487cf-10dd-4f0b-c8d5-68d9ce93e80c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using device: cpu\n"
          ]
        }
      ],
      "source": [
        "# @title\n",
        "# Install advanced augmentation and tracking libraries\n",
        "!pip install -q kagglehub albumentations mlflow\n",
        "\n",
        "import os, time, random, warnings\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from PIL import Image\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n",
        "import torchvision.models as models\n",
        "import mlflow\n",
        "\n",
        "from tqdm import tqdm\n",
        "import albumentations as A\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "import hashlib\n",
        "import kagglehub\n",
        "\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "\n",
        "# ── Reproducibility ───────────────────────────────────────────────────────────\n",
        "def seed_everything(seed: int = 42) -> None:\n",
        "    \"\"\"Pin all random number generators for full reproducibility.\"\"\"\n",
        "    random.seed(seed)\n",
        "    os.environ[\"PYTHONHASHSEED\"] = str(seed)\n",
        "    np.random.seed(seed)\n",
        "    torch.manual_seed(seed)\n",
        "    torch.cuda.manual_seed_all(seed)\n",
        "    torch.backends.cudnn.deterministic = True\n",
        "    torch.backends.cudnn.benchmark = False\n",
        "\n",
        "seed_everything(42)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Using device: {device}\")\n"
      ],
      "id": "KyDKkEHyaUUJ"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JbJqclBRaUUK"
      },
      "source": [
        "---\n",
        "## 2️⃣ Dataset Discovery\n",
        "Dynamically locate the PlantVillage directory and build a flat `DataFrame` mapping every image to its disease label.\n"
      ],
      "id": "JbJqclBRaUUK"
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rqVAunm3aUUL",
        "outputId": "7e352b0a-8395-4a02-8620-92ef15975cc3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using Colab cache for faster access to the 'plantdisease' dataset.\n",
            "Path to dataset files: /kaggle/input/plantdisease\n",
            "📂 Dataset located at: /kaggle/input/plantdisease/PlantVillage\n",
            "📊 Total images: 20,638 across 15 classes\n"
          ]
        }
      ],
      "source": [
        "# ── Download dataset via kagglehub ────────────────────────────────────────────\n",
        "path = kagglehub.dataset_download(\"emmarex/plantdisease\")\n",
        "print(\"Path to dataset files:\", path)\n",
        "\n",
        "# ── Point DATA_DIR to the PlantVillage folder ─────────────────────────────────\n",
        "DATA_DIR = os.path.join(path, \"PlantVillage\")\n",
        "print(f\"📂 Dataset located at: {DATA_DIR}\")\n",
        "\n",
        "# ── Build flat image-label dataframe ─────────────────────────────────────────\n",
        "VALID_EXTENSIONS = {\".png\", \".jpg\", \".jpeg\"}\n",
        "\n",
        "records = [\n",
        "    {\"image_path\": os.path.join(folder_path, img_file), \"disease_label\": disease_folder}\n",
        "    for disease_folder in os.listdir(DATA_DIR)\n",
        "    if os.path.isdir(folder_path := os.path.join(DATA_DIR, disease_folder))\n",
        "    for img_file in os.listdir(folder_path)\n",
        "    if os.path.splitext(img_file.lower())[1] in VALID_EXTENSIONS\n",
        "]\n",
        "\n",
        "df_all = pd.DataFrame(records)\n",
        "print(f\"📊 Total images: {len(df_all):,} across {df_all['disease_label'].nunique()} classes\")"
      ],
      "id": "rqVAunm3aUUL"
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3️⃣ DATA CLEANING"
      ],
      "metadata": {
        "id": "Cv43FapOITcH"
      },
      "id": "Cv43FapOITcH"
    },
    {
      "cell_type": "code",
      "source": [
        "# ═══════════════════════════════════════════════════════════════\n",
        "# DATA CLEANING — Check & Remove Corrupt / Blurry / Duplicate Images\n",
        "# ═══════════════════════════════════════════════════════════════\n",
        "\n",
        "issues = []\n",
        "\n",
        "def get_image_hash(img_path: str) -> str:\n",
        "    \"\"\"MD5 hash of raw file bytes — used to detect exact duplicates.\"\"\"\n",
        "    with open(img_path, \"rb\") as f:\n",
        "        return hashlib.md5(f.read()).hexdigest()\n",
        "\n",
        "def full_image_check(img_path: str, blur_threshold: float = 100.0) -> str:\n",
        "    \"\"\"\n",
        "    Run all quality checks on a single image.\n",
        "    Returns: 'clean' | 'corrupt' | 'blurry' | 'too_small'\n",
        "    \"\"\"\n",
        "    try:\n",
        "        # ── Check 1: Can PIL open it? (catches corrupt files) ────────────────\n",
        "        with Image.open(img_path) as img:\n",
        "            img.verify()  # raises if file is broken\n",
        "\n",
        "        # ── Check 2: Minimum size (rejects near-empty files) ─────────────────\n",
        "        with Image.open(img_path) as img:\n",
        "            w, h = img.size\n",
        "            if w < 32 or h < 32:\n",
        "                return \"too_small\"\n",
        "\n",
        "        # ── Check 3: Blur detection via Laplacian variance ───────────────────\n",
        "        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)\n",
        "        if gray is None:\n",
        "            return \"corrupt\"\n",
        "        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:\n",
        "            return \"blurry\"\n",
        "\n",
        "        return \"clean\"\n",
        "\n",
        "    except Exception:\n",
        "        return \"corrupt\"\n",
        "\n",
        "\n",
        "# ── Run checks on full dataset ────────────────────────────────────────────────\n",
        "print(\"Running full data quality scan …\")\n",
        "df_all[\"quality\"] = [\n",
        "    full_image_check(p)\n",
        "    for p in tqdm(df_all[\"image_path\"], desc=\"Checking images\")\n",
        "]\n",
        "\n",
        "# ── Check 4: Duplicate detection via MD5 hash ────────────────────────────────\n",
        "print(\"\\nDetecting duplicate images …\")\n",
        "df_all[\"file_hash\"] = [\n",
        "    get_image_hash(p)\n",
        "    for p in tqdm(df_all[\"image_path\"], desc=\"Hashing images\")\n",
        "]\n",
        "df_all[\"is_duplicate\"] = df_all.duplicated(subset=\"file_hash\", keep=\"first\")\n",
        "\n",
        "\n",
        "# ── Summary report ────────────────────────────────────────────────────────────\n",
        "print(\"\\n\" + \"=\"*50)\n",
        "print(\"DATA QUALITY REPORT\")\n",
        "print(\"=\"*50)\n",
        "print(f\"Total images scanned :  {len(df_all):,}\")\n",
        "print(f\"  Clean            :  {(df_all['quality'] == 'clean').sum():,}\")\n",
        "print(f\"  Corrupt          :  {(df_all['quality'] == 'corrupt').sum():,}\")\n",
        "print(f\"  Blurry           :  {(df_all['quality'] == 'blurry').sum():,}\")\n",
        "print(f\"  Too small        :  {(df_all['quality'] == 'too_small').sum():,}\")\n",
        "print(f\"  Duplicates       :  {df_all['is_duplicate'].sum():,}\")\n",
        "print(\"=\"*50)\n",
        "\n",
        "\n",
        "# ── Drop bad images from df_all ───────────────────────────────────────────────\n",
        "before = len(df_all)\n",
        "df_all = df_all[\n",
        "    (df_all[\"quality\"] == \"clean\") &\n",
        "    (df_all[\"is_duplicate\"] == False)\n",
        "].drop(columns=[\"quality\", \"file_hash\", \"is_duplicate\"]).reset_index(drop=True)\n",
        "\n",
        "removed = before - len(df_all)\n",
        "print(f\"\\nRemoved {removed:,} bad images\")\n",
        "print(f\"Clean dataset size: {len(df_all):,} images\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jGa69D_VIik-",
        "outputId": "bb6d0cc7-99e4-4504-f9ee-1cdc3f6d0045"
      },
      "id": "jGa69D_VIik-",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Running full data quality scan …\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Checking images:  50%|█████     | 10336/20638 [02:51<02:36, 66.01it/s]"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 3.1 DATASET BALANCING — Make all classes equal using augmentation"
      ],
      "metadata": {
        "id": "wUr5PNLDuA74"
      },
      "id": "wUr5PNLDuA74"
    },
    {
      "cell_type": "code",
      "source": [
        "# ═══════════════════════════════════════════════════════════════\n",
        "# 📦 DATASET BALANCING — Make all classes equal using augmentation\n",
        "# ═══════════════════════════════════════════════════════════════\n",
        "\n",
        "import os\n",
        "import cv2\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from tqdm import tqdm\n",
        "from PIL import Image as PILImage\n",
        "import albumentations as A\n",
        "\n",
        "# ── Target: make every class have this many images ────────────────────────────\n",
        "TARGET_PER_CLASS = 2200   # ← change this to whatever you want\n",
        "\n",
        "# ── Where to save the balanced dataset ───────────────────────────────────────\n",
        "BALANCED_DIR = \"/content/PlantVillage_Balanced\"\n",
        "os.makedirs(BALANCED_DIR, exist_ok=True)\n",
        "\n",
        "# ── Augmentation pipeline for generating new images ───────────────────────────\n",
        "balance_transforms = A.Compose([\n",
        "    A.Resize(224, 224),\n",
        "    A.HorizontalFlip(p=0.5),\n",
        "    A.VerticalFlip(p=0.3),\n",
        "    A.Rotate(limit=45, p=0.7),\n",
        "    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),\n",
        "    A.GaussianBlur(blur_limit=(3, 7), p=0.2),\n",
        "    A.CoarseDropout(max_holes=6, max_height=16, max_width=16, p=0.3),\n",
        "    A.Sharpen(alpha=(0.2, 0.5), p=0.3),\n",
        "    A.CLAHE(clip_limit=4.0, p=0.3),\n",
        "    A.Perspective(scale=(0.05, 0.1), p=0.3),\n",
        "    A.RandomBrightnessContrast(p=0.4),\n",
        "    A.RandomShadow(p=0.2),\n",
        "])\n",
        "\n",
        "\n",
        "def balance_class(\n",
        "    class_name    : str,\n",
        "    image_paths   : list,\n",
        "    target        : int,\n",
        "    output_dir    : str,\n",
        ") -> list:\n",
        "    \"\"\"\n",
        "    Copy all original images + generate augmented images until\n",
        "    the class reaches exactly `target` total images.\n",
        "\n",
        "    Args:\n",
        "        class_name:   Disease label string.\n",
        "        image_paths:  List of original image paths for this class.\n",
        "        target:       Target number of images.\n",
        "        output_dir:   Root output directory.\n",
        "\n",
        "    Returns:\n",
        "        List of all final image paths (original + generated).\n",
        "    \"\"\"\n",
        "    class_dir = os.path.join(output_dir, class_name)\n",
        "    os.makedirs(class_dir, exist_ok=True)\n",
        "    all_paths = []\n",
        "\n",
        "    # ── Step 1: Copy all original images ──────────────────────────────────────\n",
        "    for i, src_path in enumerate(image_paths):\n",
        "        ext      = os.path.splitext(src_path)[1].lower() or \".jpg\"\n",
        "        dst_path = os.path.join(class_dir, f\"orig_{i:04d}{ext}\")\n",
        "        if not os.path.exists(dst_path):\n",
        "            img = cv2.imread(src_path)\n",
        "            cv2.imwrite(dst_path, img)\n",
        "        all_paths.append(dst_path)\n",
        "\n",
        "    current_count = len(all_paths)\n",
        "\n",
        "    # ── Step 2: Generate augmented images to reach target ─────────────────────\n",
        "    needed = target - current_count\n",
        "    if needed <= 0:\n",
        "        # Class already has enough — just copy first `target` images\n",
        "        return all_paths[:target]\n",
        "\n",
        "    gen_idx = 0\n",
        "    while needed > 0:\n",
        "        # Pick a random source image to augment\n",
        "        src_path = image_paths[gen_idx % len(image_paths)]\n",
        "        img      = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)\n",
        "        aug      = balance_transforms(image=img)[\"image\"]\n",
        "        dst_path = os.path.join(class_dir, f\"aug_{gen_idx:05d}.jpg\")\n",
        "        cv2.imwrite(dst_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))\n",
        "        all_paths.append(dst_path)\n",
        "        gen_idx += 1\n",
        "        needed  -= 1\n",
        "\n",
        "    return all_paths\n",
        "\n",
        "\n",
        "# ── Run balancing on the full dataset ─────────────────────────────────────────\n",
        "print(f\"🎯 Target: {TARGET_PER_CLASS} images per class\")\n",
        "print(f\"📂 Output: {BALANCED_DIR}\\n\")\n",
        "\n",
        "counts        = df_all[\"disease_label\"].value_counts()\n",
        "balanced_rows = []\n",
        "\n",
        "for class_name in tqdm(counts.index, desc=\"Balancing classes\"):\n",
        "    original_paths = df_all[df_all[\"disease_label\"] == class_name][\"image_path\"].tolist()\n",
        "    original_count = len(original_paths)\n",
        "\n",
        "    final_paths = balance_class(\n",
        "        class_name  = class_name,\n",
        "        image_paths = original_paths,\n",
        "        target      = TARGET_PER_CLASS,\n",
        "        output_dir  = BALANCED_DIR,\n",
        "    )\n",
        "\n",
        "    for p in final_paths:\n",
        "        balanced_rows.append({\"image_path\": p, \"disease_label\": class_name})\n",
        "\n",
        "    action = \" kept\" if original_count >= TARGET_PER_CLASS else f\" generated {TARGET_PER_CLASS - original_count} new\"\n",
        "    print(f\"  {class_name:<45} {original_count:>5} → {TARGET_PER_CLASS}  ({action})\")\n",
        "\n",
        "\n",
        "# ── Build new balanced dataframe ──────────────────────────────────────────────\n",
        "df_balanced = pd.DataFrame(balanced_rows)\n",
        "\n",
        "print(f\"\\n{'='*55}\")\n",
        "print(f\"📊 BALANCING COMPLETE\")\n",
        "print(f\"{'='*55}\")\n",
        "print(f\"Before: {len(df_all):,} images | min={counts.min()} max={counts.max()}\")\n",
        "print(f\"After : {len(df_balanced):,} images | every class = {TARGET_PER_CLASS}\")\n",
        "print(f\"{'='*55}\")"
      ],
      "metadata": {
        "id": "FR4lDXPQuAhQ"
      },
      "id": "FR4lDXPQuAhQ",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aYTPYX2WaUUL"
      },
      "source": [
        "---\n",
        "## 3️⃣ Stratified Splitting\n",
        "\n",
        "### 3.2 Train / Val / Test Split  \n",
        "Stratified splitting (70 / 15 / 15) ensures every class has proportional representation in each subset.\n"
      ],
      "id": "aYTPYX2WaUUL"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mLlHI_cPaUUL"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ──  Stratified split ───────────────────────────────────────\n",
        "train_df, temp_df = train_test_split(\n",
        "    df_balanced, test_size=0.30, stratify=df_balanced[\"disease_label\"], random_state=42\n",
        ")\n",
        "val_df, test_df = train_test_split(\n",
        "    temp_df, test_size=0.50, stratify=temp_df[\"disease_label\"], random_state=42\n",
        ")\n",
        "\n",
        "print(f\"\\nSplits → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}\")\n"
      ],
      "id": "mLlHI_cPaUUL"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9YqG7LD7aUUL"
      },
      "source": [
        "---\n",
        "## 4️⃣ Exploratory Data Analysis (EDA)\n",
        "### 4.1 Class Distribution\n",
        "Understanding the imbalance ratio is the first step to deciding which mitigation strategy to apply.\n",
        "\n",
        "### 4.2 Sample Image Grid\n",
        "A random grid of images confirms label correctness and exposes visual variation within each class.\n"
      ],
      "id": "9YqG7LD7aUUL"
    },
    {
      "cell_type": "code",
      "source": [
        "# ── 4.1  Class distribution bar chart ────────────────────────────────────────\n",
        "fig, ax = plt.subplots(figsize=(14, 12))\n",
        "sns.countplot(\n",
        "    y=\"disease_label\", data=df_all,\n",
        "    order=df_all[\"disease_label\"].value_counts().index,\n",
        "    palette=\"crest\", ax=ax\n",
        ")\n",
        "ax.set_title(\"Before Balanced AgroMind: Image Count per Disease Class\", fontsize=18, fontweight=\"bold\", pad=15)\n",
        "ax.set_xlabel(\"Number of Images\", fontsize=13)\n",
        "ax.set_ylabel(\"Disease / Crop Class\", fontsize=13)\n",
        "for patch in ax.patches:\n",
        "    ax.annotate(\n",
        "        f\"{int(patch.get_width())}\",\n",
        "        (patch.get_width(), patch.get_y() + patch.get_height() / 2),\n",
        "        ha=\"left\", va=\"center\", xytext=(5, 0), textcoords=\"offset points\", fontsize=9\n",
        "    )\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ── Imbalance summary ─────────────────────────────────────────────────────────\n",
        "counts = df_all[\"disease_label\"].value_counts()\n",
        "print(f\"Largest class : {counts.max():,} images  ({counts.idxmax()})\")\n",
        "print(f\"Smallest class: {counts.min():,} images  ({counts.idxmin()})\")\n",
        "print(f\"Imbalance ratio: {counts.max() / counts.min():.1f}×\")\n"
      ],
      "metadata": {
        "id": "cXJ97CG0HrSm"
      },
      "id": "cXJ97CG0HrSm",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iLp0LDxaaUUL"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ── 4.1  Class distribution bar chart ────────────────────────────────────────\n",
        "fig, ax = plt.subplots(figsize=(14, 12))\n",
        "sns.countplot(\n",
        "    y=\"disease_label\", data=df_balanced,\n",
        "    order=df_balanced[\"disease_label\"].value_counts().index,\n",
        "    palette=\"crest\", ax=ax\n",
        ")\n",
        "ax.set_title(\"After Balanced AgroMind: Image Count per Disease Class\", fontsize=18, fontweight=\"bold\", pad=15)\n",
        "ax.set_xlabel(\"Number of Images\", fontsize=13)\n",
        "ax.set_ylabel(\"Disease / Crop Class\", fontsize=13)\n",
        "for patch in ax.patches:\n",
        "    ax.annotate(\n",
        "        f\"{int(patch.get_width())}\",\n",
        "        (patch.get_width(), patch.get_y() + patch.get_height() / 2),\n",
        "        ha=\"left\", va=\"center\", xytext=(5, 0), textcoords=\"offset points\", fontsize=9\n",
        "    )\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ── Imbalance summary ─────────────────────────────────────────────────────────\n",
        "counts = df_balanced[\"disease_label\"].value_counts()\n",
        "print(f\"Largest class : {counts.max():,} images  ({counts.idxmax()})\")\n",
        "print(f\"Smallest class: {counts.min():,} images  ({counts.idxmin()})\")\n",
        "print(f\"Imbalance ratio: {counts.max() / counts.min():.1f}×\")\n"
      ],
      "id": "iLp0LDxaaUUL"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vNmwEJFXaUUM"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ── 4.2  Sample image grid ────────────────────────────────────────────────────\n",
        "def visualize_sample_images(df: pd.DataFrame, num_classes: int = 5, images_per_class: int = 4) -> None:\n",
        "    \"\"\"\n",
        "    Display a random grid of images for a random subset of disease classes.\n",
        "\n",
        "    Args:\n",
        "        df:               Source dataframe with 'image_path' and 'disease_label' columns.\n",
        "        num_classes:      Number of random classes to display (rows).\n",
        "        images_per_class: Number of sample images per class (columns).\n",
        "    \"\"\"\n",
        "    sampled_classes = random.sample(\n",
        "        list(df[\"disease_label\"].unique()), min(num_classes, df[\"disease_label\"].nunique())\n",
        "    )\n",
        "    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(4 * images_per_class, 3 * num_classes))\n",
        "    fig.suptitle(\"Sample Leaf Images Across Random Disease Classes\", fontsize=17, fontweight=\"bold\", y=1.01)\n",
        "\n",
        "    for i, cls in enumerate(sampled_classes):\n",
        "        paths = df[df[\"disease_label\"] == cls][\"image_path\"].sample(images_per_class, replace=True).values\n",
        "        for j, path in enumerate(paths):\n",
        "            try:\n",
        "                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)\n",
        "                ax = axes[i, j] if num_classes > 1 else axes[j]\n",
        "                ax.imshow(img)\n",
        "                ax.axis(\"off\")\n",
        "                if j == 0:\n",
        "                    ax.set_title(\n",
        "                        cls.replace(\"___\", \": \").replace(\"_\", \" \"),\n",
        "                        fontsize=11, loc=\"left\", pad=8, fontweight=\"bold\"\n",
        "                    )\n",
        "            except Exception as e:\n",
        "                print(f\"  ⚠️  Could not load {path}: {e}\")\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "visualize_sample_images(df_all, num_classes=6, images_per_class=5)\n"
      ],
      "id": "vNmwEJFXaUUM"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "T-1_KNLyaUUM"
      },
      "source": [
        "---\n",
        "## 5️⃣ Augmentation Pipeline\n",
        "### Strategy\n",
        "| Pipeline | Used for | Augmentations |\n",
        "|----------|----------|---------------|\n",
        "| `train_transforms` | Training | Flip, Rotate, ColorJitter, GaussianBlur, Normalize |\n",
        "| `val_test_transforms` | Validation & Test | Resize only + Normalize |\n",
        "| `vis_transform` | Visual debugging | Augmentations **without** Normalize (so images render correctly) |\n",
        "\n",
        "> **Why ImageNet mean/std?** All pretrained backbones (ResNet, MobileNet, EfficientNet) were trained on ImageNet — matching its normalisation ensures the pre-learned feature weights activate correctly on new data.\n"
      ],
      "id": "T-1_KNLyaUUM"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DvVHKyDtaUUM"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "IMAGE_SIZE = 224\n",
        "\n",
        "# ── Production transforms (fed to PyTorch) ───────────────────────────────────\n",
        "train_transforms = A.Compose([\n",
        "    A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n",
        "    A.HorizontalFlip(p=0.5),\n",
        "    A.VerticalFlip(p=0.2),\n",
        "    A.Rotate(limit=30, p=0.5),\n",
        "    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),\n",
        "    A.GaussianBlur(blur_limit=(3, 7), p=0.2),\n",
        "    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # Cutout regularisation\n",
        "    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),\n",
        "    ToTensorV2()\n",
        "])\n",
        "\n",
        "val_test_transforms = A.Compose([\n",
        "    A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n",
        "    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),\n",
        "    ToTensorV2()\n",
        "])\n",
        "\n",
        "# ── Visual-only transforms (no normalise, for Matplotlib preview) ─────────────\n",
        "vis_transform = A.Compose([\n",
        "    A.Resize(IMAGE_SIZE, IMAGE_SIZE),\n",
        "    A.HorizontalFlip(p=1.0),\n",
        "    A.Rotate(limit=45, p=1.0),\n",
        "    A.ColorJitter(brightness=0.3, contrast=0.3, p=1.0),\n",
        "])\n",
        "\n",
        "# ── Before / After preview grid ───────────────────────────────────────────────\n",
        "sample_paths = train_df[\"image_path\"].sample(3, random_state=42).values\n",
        "fig, axes = plt.subplots(3, 2, figsize=(10, 12))\n",
        "fig.suptitle(\"Augmentation Effect — Original vs Augmented\", fontsize=16, fontweight=\"bold\")\n",
        "col_labels = [\"Original\", \"Augmented\"]\n",
        "for ax, label in zip(axes[0], col_labels):\n",
        "    ax.set_title(label, fontsize=14, fontweight=\"bold\")\n",
        "\n",
        "for i, path in enumerate(sample_paths):\n",
        "    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)\n",
        "    aug = vis_transform(image=img)[\"image\"]\n",
        "    for j, show in enumerate([img, aug]):\n",
        "        axes[i, j].imshow(show)\n",
        "        axes[i, j].axis(\"off\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "print(\"Augmentation pipeline defined.\")\n"
      ],
      "id": "DvVHKyDtaUUM"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EAly5tzDaUUM"
      },
      "source": [
        "---\n",
        "## 6️⃣ PyTorch Dataset & DataLoader — Imbalance Strategies\n",
        "\n",
        "Class imbalance is a critical issue in medical/agricultural datasets.  \n",
        "Three complementary strategies are implemented below:\n",
        "\n",
        "### Strategy A — WeightedRandomSampler *(applied)*\n",
        "Oversamples minority classes at the **batch-sampling level**.  \n",
        "Every class gets roughly equal representation in each batch without duplicating files on disk.\n",
        "\n",
        "### Strategy B — Class-Weighted CrossEntropyLoss *(applied as complement)*\n",
        "Applies higher loss penalties to minority-class misclassifications.  \n",
        "Pairs well with the sampler for an even stronger correction.\n",
        "\n",
        "### Strategy C — Focal Loss *(available as drop-in replacement)*\n",
        "Re-weights loss based on prediction **confidence**.  \n",
        "Hard, misclassified examples receive a larger gradient signal — especially powerful when the long-tail includes visually similar classes.\n"
      ],
      "id": "EAly5tzDaUUM"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cA45sQG_aUUM"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "# ── PyTorch Dataset ───────────────────────────────────────────────────────────\n",
        "class AgroMindDataset(Dataset):\n",
        "    \"\"\"\n",
        "    Maps image paths + string labels to (tensor, int) pairs for PyTorch.\n",
        "\n",
        "    Args:\n",
        "        dataframe:  DataFrame with columns 'image_path' and 'disease_label'.\n",
        "        transform:  Albumentations Compose pipeline.\n",
        "    \"\"\"\n",
        "    def __init__(self, dataframe: pd.DataFrame, transform=None):\n",
        "        self.df = dataframe.reset_index(drop=True)\n",
        "        self.transform = transform\n",
        "        self.labels = sorted(self.df[\"disease_label\"].unique())\n",
        "        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}\n",
        "\n",
        "    def __len__(self) -> int:\n",
        "        return len(self.df)\n",
        "\n",
        "    def __getitem__(self, idx: int):\n",
        "        img_path  = self.df.loc[idx, \"image_path\"]\n",
        "        label_str = self.df.loc[idx, \"disease_label\"]\n",
        "        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)\n",
        "        if self.transform:\n",
        "            image = self.transform(image=image)[\"image\"]\n",
        "        return image, torch.tensor(self.label_to_idx[label_str], dtype=torch.long)\n",
        "\n",
        "\n",
        "# ── Strategy A: WeightedRandomSampler ────────────────────────────────────────\n",
        "print(\"Computing per-class weights for WeightedRandomSampler …\")\n",
        "class_counts   = train_df[\"disease_label\"].value_counts().to_dict()\n",
        "num_classes    = len(class_counts)\n",
        "total_samples  = len(train_df)\n",
        "\n",
        "class_weights_dict = {\n",
        "    lbl: total_samples / (num_classes * cnt)\n",
        "    for lbl, cnt in class_counts.items()\n",
        "}\n",
        "sample_weights = train_df[\"disease_label\"].map(class_weights_dict).values\n",
        "sampler = WeightedRandomSampler(\n",
        "    weights=torch.tensor(sample_weights, dtype=torch.float),\n",
        "    num_samples=len(sample_weights),\n",
        "    replacement=True\n",
        ")\n",
        "\n",
        "\n",
        "# ── Strategy B: Class-Weighted CrossEntropyLoss ───────────────────────────────\n",
        "# Tensor of per-class weights aligned to label index order\n",
        "label_order     = sorted(class_counts.keys())\n",
        "ce_weights      = torch.tensor(\n",
        "    [class_weights_dict[lbl] for lbl in label_order], dtype=torch.float\n",
        ").to(device)\n",
        "weighted_criterion = nn.CrossEntropyLoss(weight=ce_weights)\n",
        "print(\"Weighted CrossEntropyLoss ready.\")\n",
        "\n",
        "\n",
        "# ── Strategy C: Focal Loss (drop-in for CrossEntropyLoss) ────────────────────\n",
        "class FocalLoss(nn.Module):\n",
        "    \"\"\"\n",
        "    Focal Loss — Lin et al. 2017 (https://arxiv.org/abs/1708.02002).\n",
        "\n",
        "    Down-weights well-classified examples so training focuses on hard negatives.\n",
        "    Particularly useful when the long-tail contains classes that are easy to\n",
        "    mistake for one another.\n",
        "\n",
        "    Args:\n",
        "        gamma: Focusing parameter. Higher = more focus on hard examples (typically 2.0).\n",
        "        weight: Optional per-class weight tensor (same as CrossEntropyLoss).\n",
        "    \"\"\"\n",
        "    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):\n",
        "        super().__init__()\n",
        "        self.gamma  = gamma\n",
        "        self.weight = weight\n",
        "\n",
        "    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:\n",
        "        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction=\"none\")\n",
        "        pt      = torch.exp(-ce_loss)              # probability of correct class\n",
        "        return ((1 - pt) ** self.gamma * ce_loss).mean()\n",
        "\n",
        "focal_criterion = FocalLoss(gamma=2.0, weight=ce_weights)\n",
        "print(\"Focal Loss ready (swap with weighted_criterion to activate).\")\n",
        "\n",
        "\n",
        "# ── DataLoaders ───────────────────────────────────────────────────────────────\n",
        "BATCH_SIZE = 32\n",
        "\n",
        "train_dataset = AgroMindDataset(train_df, transform=train_transforms)\n",
        "val_dataset   = AgroMindDataset(val_df,   transform=val_test_transforms)\n",
        "test_dataset  = AgroMindDataset(test_df,  transform=val_test_transforms)\n",
        "\n",
        "# shuffle=False is mandatory when using a custom sampler\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=2, pin_memory=True)\n",
        "val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,      num_workers=2, pin_memory=True)\n",
        "test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,      num_workers=2, pin_memory=True)\n",
        "\n",
        "print(f\"\\nDataLoaders ready | Batch size: {BATCH_SIZE}\")\n",
        "print(f\"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}\")\n"
      ],
      "id": "cA45sQG_aUUM"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tyJxswVUaUUN"
      },
      "source": [
        "---\n",
        "## 7️⃣ Model Architecture\n",
        "\n",
        "### BaselineCNN (custom)\n",
        "A shallow 2-block CNN used as a performance lower-bound.  \n",
        "**Overfitting mitigations added:**\n",
        "- `BatchNorm2d` — normalises activations, acts as implicit regulariser\n",
        "- `Dropout(0.5)` — randomly zeroes half the neurons before the classifier\n",
        "- Reduced hidden layer (256 → 128) to limit model capacity\n",
        "\n",
        "### Pretrained Backbones\n",
        "Transfer learning from ImageNet gives the model a strong feature prior.  \n",
        "Only the **final classification head is replaced** (fine-tuning the backbone is recommended in a second pass with a lower LR).\n"
      ],
      "id": "tyJxswVUaUUN"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Neu--9zJaUUN"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "class BaselineCNN(nn.Module):\n",
        "    \"\"\"\n",
        "    Lightweight 2-block CNN with BatchNorm and Dropout for regularisation.\n",
        "\n",
        "    Architecture overview:\n",
        "        Conv → BN → ReLU → MaxPool  (×2)\n",
        "        → Flatten → FC(256) → BN → ReLU → Dropout(0.5) → FC(num_classes)\n",
        "    \"\"\"\n",
        "    def __init__(self, num_classes: int):\n",
        "        super().__init__()\n",
        "        self.features = nn.Sequential(\n",
        "            # Block 1\n",
        "            nn.Conv2d(3, 32, kernel_size=3, padding=1),\n",
        "            nn.BatchNorm2d(32),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.MaxPool2d(2, 2),\n",
        "            # Block 2\n",
        "            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n",
        "            nn.BatchNorm2d(64),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.MaxPool2d(2, 2),\n",
        "        )\n",
        "        flat_dim = 64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)\n",
        "        self.classifier = nn.Sequential(\n",
        "            nn.Flatten(),\n",
        "            nn.Linear(flat_dim, 256),\n",
        "            nn.BatchNorm1d(256),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.Dropout(p=0.5),\n",
        "            nn.Linear(256, num_classes),\n",
        "        )\n",
        "\n",
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
        "        return self.classifier(self.features(x))\n",
        "\n",
        "\n",
        "def build_model(model_name: str, num_classes: int) -> nn.Module:\n",
        "    \"\"\"\n",
        "    Factory function: returns the requested architecture with a correctly-sized head.\n",
        "\n",
        "    Supported names: 'baseline_cnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0'\n",
        "    \"\"\"\n",
        "    if model_name == \"baseline_cnn\":\n",
        "        return BaselineCNN(num_classes)\n",
        "\n",
        "    elif model_name == \"resnet18\":\n",
        "        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n",
        "        model.fc = nn.Sequential(\n",
        "            nn.Dropout(p=0.3),\n",
        "            nn.Linear(model.fc.in_features, num_classes)\n",
        "        )\n",
        "        return model\n",
        "\n",
        "    elif model_name == \"mobilenet_v2\":\n",
        "        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)\n",
        "        model.classifier[1] = nn.Sequential(\n",
        "            nn.Dropout(p=0.3),\n",
        "            nn.Linear(model.classifier[1].in_features, num_classes)\n",
        "        )\n",
        "        return model\n",
        "\n",
        "    elif model_name == \"efficientnet_b0\":\n",
        "        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)\n",
        "        model.classifier[1] = nn.Sequential(\n",
        "            nn.Dropout(p=0.3),\n",
        "            nn.Linear(model.classifier[1].in_features, num_classes)\n",
        "        )\n",
        "        return model\n",
        "\n",
        "    else:\n",
        "        raise ValueError(f\"Unknown model name '{model_name}'. \"\n",
        "                         \"Choose from: baseline_cnn, resnet18, mobilenet_v2, efficientnet_b0\")\n",
        "\n",
        "\n",
        "# Quick sanity check\n",
        "_test_model = build_model(\"efficientnet_b0\", num_classes).to(device)\n",
        "_dummy      = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)\n",
        "print(f\" EfficientNet-B0 forward pass shape: {_test_model(_dummy).shape}\")\n",
        "del _test_model, _dummy\n"
      ],
      "id": "Neu--9zJaUUN"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QMfpSauyaUUN"
      },
      "source": [
        "---\n",
        "## 8️⃣ Training Loop — Overfitting Prevention Strategies\n",
        "\n",
        "### Strategies implemented\n",
        "\n",
        "| Strategy | Mechanism | Where applied |\n",
        "|----------|-----------|---------------|\n",
        "| **Dropout** | Randomly zero neurons during training | In model heads (§7) |\n",
        "| **BatchNorm** | Normalise activations, implicit regulariser | BaselineCNN blocks |\n",
        "| **Cutout / CoarseDropout** | Randomly mask image patches | In `train_transforms` (§5) |\n",
        "| **Weight Decay (L2)** | Penalises large weights in the optimizer | `AdamW(weight_decay=1e-4)` |\n",
        "| **LR Scheduling** | Reduce LR when val loss plateaus | `ReduceLROnPlateau` |\n",
        "| **Early Stopping** | Halt when val accuracy stops improving | `patience` parameter |\n",
        "| **Best-model Checkpointing** | Save only the best-val-acc weights | `torch.save` on improvement |\n",
        "\n",
        "> **Tip:** The most impactful single addition is Early Stopping + Checkpointing — it prevents the final epoch from being a degraded model instead of the best one.\n"
      ],
      "id": "QMfpSauyaUUN"
    },
    {
      "cell_type": "code",
      "source": [
        "# ── Early Stopping helper ─────────────────────────────────────────────────────\n",
        "class EarlyStopping:\n",
        "    \"\"\"\n",
        "    Monitors validation accuracy and signals when training should stop.\n",
        "\n",
        "    Args:\n",
        "        patience:  Epochs to wait for improvement before stopping.\n",
        "        min_delta: Minimum improvement to count as 'better'.\n",
        "    \"\"\"\n",
        "    def __init__(self, patience: int = 5, min_delta: float = 1e-4):\n",
        "        self.patience   = patience\n",
        "        self.min_delta  = min_delta\n",
        "        self.best_score = None\n",
        "        self.counter    = 0\n",
        "        self.stop       = False\n",
        "\n",
        "    def __call__(self, val_acc: float) -> bool:\n",
        "        if self.best_score is None or val_acc > self.best_score + self.min_delta:\n",
        "            self.best_score = val_acc\n",
        "            self.counter    = 0\n",
        "        else:\n",
        "            self.counter += 1\n",
        "            if self.counter >= self.patience:\n",
        "                self.stop = True\n",
        "        return self.stop"
      ],
      "metadata": {
        "id": "5I9Ei2XwDBEb"
      },
      "id": "5I9Ei2XwDBEb",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JCFGqYHCaUUN"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "\n",
        "def train_model(\n",
        "    model_name  : str,\n",
        "    epochs      : int   = 15,\n",
        "    lr          : float = 1e-4,\n",
        "    weight_decay: float = 1e-4,\n",
        "    patience    : int   = 5,\n",
        "    use_focal   : bool  = False,\n",
        ") -> tuple[str, dict]:\n",
        "\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"  Training: {model_name.upper()}\")\n",
        "    print(f\"{'='*60}\")\n",
        "\n",
        "    model      = build_model(model_name, num_classes).to(device)\n",
        "    criterion  = focal_criterion if use_focal else weighted_criterion\n",
        "    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)\n",
        "    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\"min\", factor=0.5, patience=3)\n",
        "    early_stop = EarlyStopping(patience=patience)\n",
        "\n",
        "    SAVE_DIR  = \"/kaggle/working\" if os.path.exists(\"/kaggle/working\") else \"/content\"\n",
        "    save_path = f\"{SAVE_DIR}/{model_name}_best.pth\"\n",
        "    history   = {\"train_loss\": [], \"val_loss\": [], \"train_acc\": [], \"val_acc\": []}\n",
        "    best_val_acc = 0.0\n",
        "\n",
        "    mlflow.set_experiment(\"AgroMind_Benchmarking\")\n",
        "    with mlflow.start_run(run_name=model_name):\n",
        "        mlflow.log_params({\n",
        "            \"architecture\": model_name, \"epochs\": epochs,\n",
        "            \"batch_size\": BATCH_SIZE, \"lr\": lr,\n",
        "            \"weight_decay\": weight_decay, \"loss\": \"focal\" if use_focal else \"weighted_ce\",\n",
        "        })\n",
        "\n",
        "        for epoch in range(1, epochs + 1):\n",
        "\n",
        "            # ── Train ──────────────────────────────────────────────────────────\n",
        "            model.train()\n",
        "            running_loss = correct = total = 0\n",
        "\n",
        "            train_bar = tqdm(train_loader, desc=f\"Epoch {epoch:>2}/{epochs} [Train]\", leave=False)\n",
        "            for images, labels in train_bar:\n",
        "                images, labels = images.to(device), labels.to(device)\n",
        "                optimizer.zero_grad()\n",
        "                outputs = model(images)\n",
        "                loss    = criterion(outputs, labels)\n",
        "                loss.backward()\n",
        "                optimizer.step()\n",
        "\n",
        "                running_loss += loss.item() * images.size(0)\n",
        "                _, predicted  = outputs.max(1)\n",
        "                total        += labels.size(0)\n",
        "                correct      += predicted.eq(labels).sum().item()\n",
        "\n",
        "                train_bar.set_postfix({\n",
        "                    \"loss\": f\"{running_loss / total:.4f}\",\n",
        "                    \"acc\" : f\"{correct / total:.4f}\"\n",
        "                })\n",
        "\n",
        "            train_loss = running_loss / total\n",
        "            train_acc  = correct / total\n",
        "\n",
        "            # ── Validate ───────────────────────────────────────────────────────\n",
        "            model.eval()\n",
        "            val_running_loss = val_correct = val_total = 0\n",
        "\n",
        "            val_bar = tqdm(val_loader, desc=f\"Epoch {epoch:>2}/{epochs} [Val]  \", leave=False)\n",
        "            with torch.no_grad():\n",
        "                for images, labels in val_bar:\n",
        "                    images, labels = images.to(device), labels.to(device)\n",
        "                    outputs  = model(images)\n",
        "                    loss     = criterion(outputs, labels)\n",
        "                    val_running_loss += loss.item() * images.size(0)\n",
        "                    _, predicted      = outputs.max(1)\n",
        "                    val_total        += labels.size(0)\n",
        "                    val_correct      += predicted.eq(labels).sum().item()\n",
        "\n",
        "                    val_bar.set_postfix({\n",
        "                        \"loss\": f\"{val_running_loss / val_total:.4f}\",\n",
        "                        \"acc\" : f\"{val_correct / val_total:.4f}\"\n",
        "                    })\n",
        "\n",
        "            val_loss = val_running_loss / val_total\n",
        "            val_acc  = val_correct / val_total\n",
        "\n",
        "            # ── Log & record ──────────────────────────────────────────────────\n",
        "            history[\"train_loss\"].append(train_loss)\n",
        "            history[\"val_loss\"].append(val_loss)\n",
        "            history[\"train_acc\"].append(train_acc)\n",
        "            history[\"val_acc\"].append(val_acc)\n",
        "            mlflow.log_metrics(\n",
        "                {\"train_loss\": train_loss, \"val_loss\": val_loss,\n",
        "                 \"train_acc\": train_acc,   \"val_acc\": val_acc},\n",
        "                step=epoch\n",
        "            )\n",
        "\n",
        "            current_lr = optimizer.param_groups[0][\"lr\"]\n",
        "            print(\n",
        "                f\"Epoch {epoch:>3}/{epochs} | \"\n",
        "                f\"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | \"\n",
        "                f\"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | LR: {current_lr:.2e}\"\n",
        "            )\n",
        "\n",
        "            # ── Checkpoint ────────────────────────────────────────────────────\n",
        "            if val_acc > best_val_acc:\n",
        "                best_val_acc = val_acc\n",
        "                torch.save(model.state_dict(), save_path)\n",
        "                print(f\"   Saved best model → {save_path}  (val_acc={val_acc:.4f})\")\n",
        "\n",
        "            # ── LR scheduler ──────────────────────────────────────────────────\n",
        "            scheduler.step(val_loss)\n",
        "\n",
        "            # ── Early stopping ────────────────────────────────────────────────\n",
        "            if early_stop(val_acc):\n",
        "                print(f\"    Early stopping after {epoch} epochs.\")\n",
        "                break\n",
        "\n",
        "        mlflow.log_metric(\"best_val_acc\", best_val_acc)\n",
        "        print(f\"\\n Best val acc: {best_val_acc:.4f} — saved at {save_path}\")\n",
        "\n",
        "    return save_path, history"
      ],
      "id": "JCFGqYHCaUUN"
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def plot_training_history(history, model_name=\"Model\"):\n",
        "    \"\"\"\n",
        "    Plots the training and validation loss and accuracy from the history dictionary.\n",
        "    \"\"\"\n",
        "    epochs = range(1, len(history['train_loss']) + 1)\n",
        "\n",
        "    plt.figure(figsize=(14, 5))\n",
        "\n",
        "    # 1. Plot Loss\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')\n",
        "    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')\n",
        "    plt.title(f'{model_name} - Training & Validation Loss')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('Loss')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "\n",
        "    # 2. Plot Accuracy\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')\n",
        "    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')\n",
        "    plt.title(f'{model_name} - Training & Validation Accuracy')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('Accuracy')\n",
        "    plt.legend()\n",
        "    plt.grid(True)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.show()"
      ],
      "metadata": {
        "id": "FX-fUZkbIyOC"
      },
      "id": "FX-fUZkbIyOC",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n7BlTcg-aUUO"
      },
      "source": [
        "### 8.1 Train EfficientNet-B0"
      ],
      "id": "n7BlTcg-aUUO"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZQTo-d_LaUUO"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "efficientnet_path, efficientnet_history = train_model(\"efficientnet_b0\", epochs=15, patience=3, use_focal=True)\n",
        "plot_training_history(efficientnet_history, \"EfficientNet-B0\")\n"
      ],
      "id": "ZQTo-d_LaUUO"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dPGkmz__aUUO"
      },
      "source": [
        "### 8.2 Train MobileNet-V2"
      ],
      "id": "dPGkmz__aUUO"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aSECydJoaUUO"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "mobilenet_path, mobilenet_history = train_model(\"mobilenet_v2\", epochs=15, patience=3,use_focal=True )\n",
        "plot_training_history(mobilenet_history, \"MobileNet-V2\")\n"
      ],
      "id": "aSECydJoaUUO"
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 8.3 Train resnet18"
      ],
      "metadata": {
        "id": "N23voSar_v7s"
      },
      "id": "N23voSar_v7s"
    },
    {
      "cell_type": "code",
      "source": [
        "resnet_path, resnet_history = train_model(\"resnet18\", epochs=15, patience=3,use_focal=True)\n",
        "plot_training_history(resnet_history, \"resnet18-V2\")\n"
      ],
      "metadata": {
        "id": "ERUCvaKf_nCt"
      },
      "id": "ERUCvaKf_nCt",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "po-BLYUzaUUO"
      },
      "source": [
        "---\n",
        "## 9️⃣ Model Evaluation\n",
        "Each model is loaded from its **best checkpoint** (not the final epoch — this is key when early stopping is active).  \n",
        "Outputs:\n",
        "- Full `classification_report` (precision / recall / F1 per class + macro/weighted averages)\n",
        "- Confusion matrix heatmap\n"
      ],
      "id": "po-BLYUzaUUO"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "12e0j8E9aUUO"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "def evaluate_model(model_name: str, model_path: str, loader: DataLoader, class_names: list) -> None:\n",
        "    \"\"\"\n",
        "    Load the best checkpoint and compute classification metrics on a given loader.\n",
        "\n",
        "    Args:\n",
        "        model_name:   Architecture key used to rebuild the model skeleton.\n",
        "        model_path:   Path to the saved .pth weights file.\n",
        "        loader:       DataLoader for the evaluation split (val or test).\n",
        "        class_names:  Ordered list of class label strings.\n",
        "    \"\"\"\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"  Evaluating: {model_name.upper()}\")\n",
        "    print(f\"{'='*60}\")\n",
        "\n",
        "    model = build_model(model_name, num_classes).to(device)\n",
        "    model.load_state_dict(torch.load(model_path, map_location=device))\n",
        "    model.eval()\n",
        "\n",
        "    all_preds, all_labels = [], []\n",
        "    with torch.no_grad():\n",
        "        for images, labels in loader:\n",
        "            images = images.to(device)\n",
        "            _, predicted = model(images).max(1)\n",
        "            all_preds.extend(predicted.cpu().numpy())\n",
        "            all_labels.extend(labels.numpy())\n",
        "\n",
        "    print(\"\\n📋 Classification Report:\")\n",
        "    print(classification_report(all_labels, all_preds, target_names=class_names))\n",
        "\n",
        "    cm = confusion_matrix(all_labels, all_preds)\n",
        "    fig, ax = plt.subplots(figsize=(14, 12))\n",
        "    sns.heatmap(\n",
        "        cm, annot=False, cmap=\"Blues\",\n",
        "        xticklabels=class_names, yticklabels=class_names, ax=ax\n",
        "    )\n",
        "    ax.set_title(f\"Confusion Matrix — {model_name}\", fontsize=16, fontweight=\"bold\")\n",
        "    ax.set_ylabel(\"True Class\"); ax.set_xlabel(\"Predicted Class\")\n",
        "    plt.xticks(rotation=90); plt.tight_layout(); plt.show()\n"
      ],
      "id": "12e0j8E9aUUO"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tcsMh286aUUO"
      },
      "source": [
        "### 9.1 Evaluate EfficientNet-B0"
      ],
      "id": "tcsMh286aUUO"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A8hU2dVuaUUO"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "evaluate_model('efficientnet_b0', efficientnet_path, test_loader, train_dataset.labels)"
      ],
      "id": "A8hU2dVuaUUO"
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U3DjbLlvaUUO"
      },
      "source": [
        "### 9.2 Evaluate MobileNet-V2"
      ],
      "id": "U3DjbLlvaUUO"
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MWxvVs8JaUUP"
      },
      "execution_count": null,
      "outputs": [],
      "source": [
        "evaluate_model('mobilenet_v2', mobilenet_path, test_loader, train_dataset.labels)"
      ],
      "id": "MWxvVs8JaUUP"
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 9.3 Evaluate resnet18"
      ],
      "metadata": {
        "id": "DbXJS3f5ATf7"
      },
      "id": "DbXJS3f5ATf7"
    },
    {
      "cell_type": "code",
      "source": [
        "evaluate_model('resnet18', resnet_path, test_loader, train_dataset.labels)"
      ],
      "metadata": {
        "id": "xkWFWPxeAfwn"
      },
      "id": "xkWFWPxeAfwn",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 9.4 AUTOMATED MODEL COMPARISON & SELECTION"
      ],
      "metadata": {
        "id": "bx0nDRPeYzka"
      },
      "id": "bx0nDRPeYzka"
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================================================================\n",
        "# 9.5 AUTOMATED MODEL COMPARISON & SELECTION\n",
        "# ==============================================================================\n",
        "\n",
        "# 1. Group the paths and histories generated from sections 8.1, 8.2, and 8.3\n",
        "models_data = {\n",
        "    \"efficientnet_b0\": {\"path\": efficientnet_path, \"history\": efficientnet_history},\n",
        "    \"mobilenet_v2\": {\"path\": mobilenet_path, \"history\": mobilenet_history},\n",
        "    \"resnet18\": {\"path\": resnet_path, \"history\": resnet_history}\n",
        "}\n",
        "\n",
        "best_model_name = None\n",
        "best_val_acc = 0.0\n",
        "best_model_path = \"\"\n",
        "\n",
        "print(\" MODEL PERFORMANCE COMPARISON\")\n",
        "print(\"=\" * 45)\n",
        "\n",
        "# 2. Iterate through the data to find the highest accuracy and lowest loss\n",
        "for arch_name, data in models_data.items():\n",
        "    # Extract the peak accuracy and minimum loss from the history dictionaries\n",
        "    max_acc = max(data[\"history\"][\"val_acc\"])\n",
        "    min_loss = min(data[\"history\"][\"val_loss\"])\n",
        "\n",
        "    print(f\"{arch_name.upper()}:\")\n",
        "    print(f\"  ▶ Peak Validation Accuracy : {max_acc:.4f}\")\n",
        "    print(f\"  ▶ Best Validation Loss     : {min_loss:.4f}\\n\")\n",
        "\n",
        "    # Update the winner if this model's accuracy is the highest so far\n",
        "    if max_acc > best_val_acc:\n",
        "        best_val_acc = max_acc\n",
        "        best_model_name = arch_name\n",
        "        best_model_path = data[\"path\"]\n",
        "\n",
        "print(\"=\" * 45)\n",
        "print(f\" AUTOMATIC WINNER: {best_model_name.upper()}\")\n",
        "print(f\"Achieved highest accuracy of {best_val_acc:.4f}.\")\n",
        "print(f\"Path saved at: {best_model_path}\\n\")\n",
        "\n",
        "# 3. Load the winning model dynamically for the production pipeline\n",
        "print(f\"Loading {best_model_name.upper()} into production memory...\")\n",
        "prod_model = build_model(best_model_name, num_classes).to(device)\n",
        "prod_model.load_state_dict(torch.load(best_model_path, map_location=device))\n",
        "print(\" Production model ready for the GenAI Gatekeeper!\")"
      ],
      "metadata": {
        "id": "Ex0uc56eAm6G"
      },
      "id": "Ex0uc56eAm6G",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 10 Streamlit"
      ],
      "metadata": {
        "id": "2nl6wdNvMcVk"
      },
      "id": "2nl6wdNvMcVk"
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit albumentations torch torchvision opencv-python"
      ],
      "metadata": {
        "id": "K5LBBS69mKR0",
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "84b93a20-573b-4a69-dcaa-8f5b6d87aeaa"
      },
      "id": "K5LBBS69mKR0",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.58.0-py3-none-any.whl.metadata (9.6 kB)\n",
            "Requirement already satisfied: albumentations in /usr/local/lib/python3.12/dist-packages (2.0.8)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.11.0+cpu)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.12/dist-packages (0.26.0+cpu)\n",
            "Requirement already satisfied: opencv-python in /usr/local/lib/python3.12/dist-packages (4.13.0.92)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<8,>=5.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.2.6)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.4.1)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.50)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (26.2)\n",
            "Requirement already satisfied: pandas<4,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.2-py2.py3-none-any.whl.metadata (4.2 kB)\n",
            "Requirement already satisfied: protobuf<8,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.29.6)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (9.1.4)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)\n",
            "Requirement already satisfied: starlette>=0.40.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.52.1)\n",
            "Requirement already satisfied: uvicorn>=0.30.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.49.0)\n",
            "Requirement already satisfied: httptools>=0.6.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.8.0)\n",
            "Requirement already satisfied: anyio>=4.0.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.13.0)\n",
            "Requirement already satisfied: python-multipart>=0.0.10 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.0.32)\n",
            "Requirement already satisfied: websockets>=12.0.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (15.0.1)\n",
            "Requirement already satisfied: itsdangerous>=2.1.2 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: scipy>=1.10.0 in /usr/local/lib/python3.12/dist-packages (from albumentations) (1.16.3)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.12/dist-packages (from albumentations) (6.0.3)\n",
            "Requirement already satisfied: pydantic>=2.9.2 in /usr/local/lib/python3.12/dist-packages (from albumentations) (2.12.3)\n",
            "Requirement already satisfied: albucore==0.0.24 in /usr/local/lib/python3.12/dist-packages (from albumentations) (0.0.24)\n",
            "Requirement already satisfied: opencv-python-headless>=4.9.0.80 in /usr/local/lib/python3.12/dist-packages (from albumentations) (4.13.0.92)\n",
            "Requirement already satisfied: stringzilla>=3.10.4 in /usr/local/lib/python3.12/dist-packages (from albucore==0.0.24->albumentations) (4.6.2)\n",
            "Requirement already satisfied: simsimd>=5.9.2 in /usr/local/lib/python3.12/dist-packages (from albucore==0.0.24->albumentations) (6.5.16)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.29.3)\n",
            "Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.26.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.22.1)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.12/dist-packages (from anyio>=4.0.0->streamlit) (3.18)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<4,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<4,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<4,>=1.4.0->streamlit) (2026.2)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.9.2->albumentations) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.9.2->albumentations) (2.41.4)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.9.2->albumentations) (0.4.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2026.5.20)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.12/dist-packages (from uvicorn>=0.30.0->streamlit) (0.16.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (26.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2026.5.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<4,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.58.0-py3-none-any.whl (9.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.2/9.2 MB\u001b[0m \u001b[31m63.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.2-py2.py3-none-any.whl (11.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.3/11.3 MB\u001b[0m \u001b[31m74.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.2 streamlit-1.58.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ls /content/"
      ],
      "metadata": {
        "id": "fBkVNsH2n32y",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3cf337ed-7c8c-4e66-ff37-5216bd93a4c0"
      },
      "id": "fBkVNsH2n32y",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "biolens_app.py  efficientnet_b0_best.pth  \u001b[0m\u001b[01;34msample_data\u001b[0m/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pyngrok -q"
      ],
      "metadata": {
        "id": "MeBS7Rcph-uC"
      },
      "id": "MeBS7Rcph-uC",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "\n",
        "CLASS_NAMES = sorted([\n",
        "    \"Apple___Apple_scab\",\"Apple___Black_rot\",\"Apple___Cedar_apple_rust\",\"Apple___healthy\",\n",
        "    \"Blueberry___healthy\",\"Cherry_(including_sour)___Powdery_mildew\",\n",
        "    \"Cherry_(including_sour)___healthy\",\"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot\",\n",
        "    \"Corn_(maize)___Common_rust_\",\"Corn_(maize)___Northern_Leaf_Blight\",\"Corn_(maize)___healthy\",\n",
        "    \"Grape___Black_rot\",\"Grape___Esca_(Black_Measles)\",\"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)\",\n",
        "    \"Grape___healthy\",\"Orange___Haunglongbing_(Citrus_greening)\",\n",
        "    \"Peach___Bacterial_spot\",\"Peach___healthy\",\"Pepper,_bell___Bacterial_spot\",\n",
        "    \"Pepper,_bell___healthy\",\"Potato___Early_blight\",\"Potato___Late_blight\",\"Potato___healthy\",\n",
        "    \"Raspberry___healthy\",\"Soybean___healthy\",\"Squash___Powdery_mildew\",\n",
        "    \"Strawberry___Leaf_scorch\",\"Strawberry___healthy\",\"Tomato___Bacterial_spot\",\n",
        "    \"Tomato___Early_blight\",\"Tomato___Late_blight\",\"Tomato___Leaf_Mold\",\n",
        "    \"Tomato___Septoria_leaf_spot\",\"Tomato___Spider_mites Two-spotted_spider_mite\",\n",
        "    \"Tomato___Target_Spot\",\"Tomato___Tomato_Yellow_Leaf_Curl_Virus\",\n",
        "    \"Tomato___Tomato_mosaic_virus\",\"Tomato___healthy\",\n",
        "])\n",
        "\n",
        "with open(\"/content/class_names.json\", \"w\") as f:\n",
        "    json.dump(CLASS_NAMES, f)\n",
        "\n",
        "print(\"✅ class_names.json saved:\", len(CLASS_NAMES), \"classes\")"
      ],
      "metadata": {
        "id": "VE-Q3vjBl7vQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "65ec3295-5d2b-47e4-a4e6-40fda549790e"
      },
      "id": "VE-Q3vjBl7vQ",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ class_names.json saved: 38 classes\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fix = '''\n",
        "def validate_leaf(img_pil):\n",
        "    try:\n",
        "        client = OpenAI(base_url=\"https://openrouter.ai/api/v1\", api_key=OPENROUTER_API_KEY)\n",
        "        resp = client.chat.completions.create(\n",
        "            model=\"openai/gpt-4o-mini\",\n",
        "            messages=[{\"role\":\"user\",\"content\":[\n",
        "                {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,\"+img_to_b64(img_pil)}},\n",
        "                {\"type\":\"text\",\"text\":(\n",
        "                    \"Is this a plant leaf or any part of a plant?\\\\n\"\n",
        "                    \"Answer EXACTLY:\\\\n\"\n",
        "                    \"ANSWER: YES or NO\\\\n\"\n",
        "                    \"REASON: one sentence\\\\n\"\n",
        "                    \"SUBJECT: what you see\"\n",
        "                )},\n",
        "            ]}],\n",
        "            max_tokens=150,\n",
        "        )\n",
        "        text = resp.choices[0].message.content.strip()\n",
        "        is_leaf = \"ANSWER: YES\" in text.upper()\n",
        "        reason = subject = \"\"\n",
        "        for line in text.split(\"\\\\n\"):\n",
        "            if line.upper().startswith(\"REASON:\"): reason = line.split(\":\",1)[1].strip()\n",
        "            if line.upper().startswith(\"SUBJECT:\"): subject = line.split(\":\",1)[1].strip()\n",
        "        explanation = f\"{subject} — {reason}\" if subject else reason\n",
        "        return is_leaf, explanation\n",
        "    except Exception as e:\n",
        "        return True, str(e)\n",
        "'''\n",
        "\n",
        "\n",
        "with open(\"/content/biolens_app.py\", \"r\") as f:\n",
        "    content = f.read()\n",
        "\n",
        "\n",
        "import re\n",
        "content = re.sub(\n",
        "    r'def validate_leaf\\(img_pil\\):.*?(?=\\n# ══|$)',\n",
        "    fix.strip(),\n",
        "    content,\n",
        "    flags=re.DOTALL\n",
        ")\n",
        "\n",
        "with open(\"/content/biolens_app.py\", \"w\") as f:\n",
        "    f.write(content)\n",
        "\n",
        "print(\"✅ Fixed! Now restart Streamlit\")"
      ],
      "metadata": {
        "id": "Ygda5xe0mome",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c88a9a4a-1386-41ed-e4e0-00347556f5f7"
      },
      "id": "Ygda5xe0mome",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Fixed! Now restart Streamlit\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import subprocess, time\n",
        "\n",
        "# ثبّت المكتبات\n",
        "subprocess.run([\"pip\", \"install\", \"streamlit\", \"pyngrok\", \"openai\", \"-q\"])\n",
        "\n",
        "# جيب الملفات من Drive\n",
        "from google.colab import drive\n",
        "# Updated mount path\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Updated file transfer paths\n",
        "subprocess.run([\"cp\", \"/content/drive/MyDrive/biolens_app.py\", \"/content/biolens_app.py\"])\n",
        "subprocess.run([\"cp\", \"/content/drive/MyDrive/class_names.json\", \"/content/class_names.json\"])\n",
        "subprocess.run([\"rsync\", \"-ah\", \"/content/drive/MyDrive/efficientnet_b0_best.pth\", \"/content/efficientnet_b0_best.pth\"])\n",
        "\n",
        "print(\"✅\" )"
      ],
      "metadata": {
        "id": "-ZkzI5Z-tXjI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3c00199f-b438-42ff-ef61-806f3238112b"
      },
      "id": "-ZkzI5Z-tXjI",
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n",
            "✅ جاهز - شغل الخلية الثانية\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import subprocess, time\n",
        "from pyngrok import ngrok\n",
        "\n",
        "ngrok.set_auth_token(\"YOUR_NGROK_AUTH_TOKEN\")\n",
        "ngrok.kill()\n",
        "time.sleep(2)\n",
        "\n",
        "subprocess.Popen([\"streamlit\", \"run\", \"/content/biolens_app.py\", \"--server.port\", \"8501\", \"--server.headless\", \"true\"])\n",
        "time.sleep(8)\n",
        "\n",
        "url = ngrok.connect(8501)\n",
        "print(\"🌿 BioLens AI:\", url)"
      ],
      "metadata": {
        "id": "Da9xAj2otfG-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "90e4b1b3-e439-42b2-93a8-6b4b3c255de3"
      },
      "id": "Da9xAj2otfG-",
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "🌿 BioLens AI: NgrokTunnel: \"https://prelaunch-crinkly-payee.ngrok-free.dev\" -> \"http://localhost:8501\"\n"
          ]
        }
      ]
    }
  ]
}