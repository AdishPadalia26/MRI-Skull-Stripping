# MRI Skull-Stripping with Shift-Invariant Modified U-Net

Lightweight, robust brain MRI skull-stripping using a **Modified U-Net** with **Adaptive Polyphase Pooling (APP)** layers, ensuring shift-invariance and stable predictions across translations in the imaging plane.

---

## 🧠 Abstract

Skull-stripping is a crucial preprocessing step in medical image analysis, separating brain tissue from non-brain regions in MRI scans. This project introduces a **Modified U-Net architecture with Adaptive Polyphase Pooling (APP)** to achieve **shift-invariance**, addressing the inconsistency of traditional CNNs under input shifts.  

Adaptive Polyphase Pooling ensures that the model produces consistent segmentation outputs, even if the MRI slices are shifted in-plane, making the method more **robust and reliable** for downstream tasks such as registration, volumetric measurement, and disease analysis.  

Compared to traditional methods like **FreeSurfer, FSL, and CAT**, our approach is **faster (2–3 mins per volume vs 15–20 mins)** and achieves **state-of-the-art accuracy (99.2%)**.

📄 **Research Paper:**  
[Enhancing Shift-Invariance for Accurate Brain MRI Skull-Stripping using Adaptive Polyphase Pooling in Modified U-Net (IEEE ICACRS 2023)](https://doi.org/10.1109/ICACRS58579.2023.10404359)  
**Authors:** A. Padalia, T. Shah, P. Pujari, A. Karande

---

## 📁 Repository Structure

```
.
├─ MRI_Skull_Stripping.ipynb       # End-to-end notebook (preprocessing → training → inference → visualization)
├─ MRI_Visualiser/                 # Visualization tools (scroll through slices, overlays, save outputs)
└─ Enhancing_Shift-Invariance...pdf# Published research paper (reference)
```

---

## ⚙️ Requirements

- Python ≥ 3.9  
- TensorFlow / Keras (preferred) or PyTorch  
- NumPy, SciPy, scikit-image, scikit-learn  
- nibabel (NIfTI support)  
- OpenCV, matplotlib, seaborn, tqdm  

Install with:

```bash
# create environment
conda create -n mri-strip python=3.10 -y
conda activate mri-strip

# install core packages
pip install numpy scipy scikit-image scikit-learn nibabel opencv-python matplotlib seaborn tqdm
pip install tensorflow==2.*  # or torch torchvision torchaudio
```

---

## 📦 Dataset

We trained and tested on the Internet Brain Segmentation Repository (IBSR) dataset:

- 18 T1-weighted MRI brain scans (ages 20–30)
- Manually segmented masks of 43 subcortical structures (hippocampus, amygdala, thalamus, etc.)

Data format:

```
data/
├─ subject_001/
│  ├─ t1.nii.gz
│  └─ mask.nii.gz   # ground-truth mask
└─ subject_002/
   ├─ t1.nii.gz
   └─ mask.nii.gz
```

---

## 🚀 Quick Start

Clone the repo:

```bash
git clone https://github.com/AdishPadalia26/MRI-Skull-Stripping.git
cd MRI-Skull-Stripping
```

Setup the environment (see above).

Launch Jupyter:

```bash
jupyter lab
```

Run `MRI_Skull_Stripping.ipynb`:

- Update dataset paths
- Execute all cells → preprocessing → training/inference → visualization

---

## 🧩 Methodology

- **Input:** 3D volumetric MRI (T1-weighted), resized to 256×256×256 and oriented to match ground-truth mask
- **Network:** Modified U-Net with added skip connections and Adaptive Polyphase Pooling layers
- **Encoder:** extracts multi-resolution features (8×8 → 256×256)
- **Adaptive Polyphase Pooling downsampling:** replaces stride-based pooling, ensuring shift-invariance
- **Decoder:** reconstructs mask with Adaptive Polyphase Pooling upsampling
- **Activation:** ReLU, with zero-padding
- **Loss:** Pixel-wise segmentation with Dice & IoU metrics
- **Batch size:** 32

🖼️ Architecture overview (from paper):  
Modified U-Net with Adaptive Polyphase Pooling ensures robust segmentation, even when slices are shifted.

---

## 📊 Results

### 1. Shift-Invariance Analysis

We tested the model under **pixel shifts along x and y axes**.  

- **Without Adaptive Polyphase Pooling (Table III):** accuracy fluctuates significantly depending on the shift, sometimes dropping as low as **74%** or **79%**.  
- **With Adaptive Polyphase Pooling (Table IV):** accuracy remains stable, consistently around **99.2%–99.4%**, confirming **shift-invariance**.

#### Table III: List of accuracies on U-Net without Adaptive Polyphase Pooling Layers

| Pixel Range | 0     | 64    | 128   | 192   | 256   |
|-------------|-------|-------|-------|-------|-------|
| **0**       | 99.21 | 98.23 | 98.27 | 98.21 | 98.19 |
| **64**      | 74.32 | 98.45 | 97.13 | 88.23 | 91.22 |
| **128**     | 96.34 | 92.1  | 98.49 | 99.1  | 93.2  |
| **192**     | 98.23 | 97.56 | 79.34 | 94    | 98    |
| **256**     | 94.9  | 98.65 | 95.55 | 96.23 | 94.13 |

#### Table IV: List of accuracies on U-Net with Adaptive Polyphase Pooling Layers

| Pixel Range | 0     | 64    | 128   | 192   | 256   |
|-------------|-------|-------|-------|-------|-------|
| **0**       | 99.27 | 99.26 | 99.27 | 99.21 | 99.19 |
| **64**      | 99.35 | 99.45 | 99.2  | 99.37 | 99.37 |
| **128**     | 99.41 | 99.37 | 99.19 | 99.39 | 99.35 |
| **192**     | 99.1  | 99    | 99.28 | 99.3  | 99.22 |
| **256**     | 99.33 | 99.39 | 99.46 | 99.39 | 99.35 |

![Shift-Invariance Results](image1)

📌 **Key Insight:** Adaptive Polyphase Pooling ensures consistent accuracy across all displacements, unlike standard pooling which degrades performance under shifts.

---

### 2. Comparison with Existing Tools

| Method                                | Time per volume | Accuracy | Efficiency |
|----------------------------------------|-----------------|----------|------------|
| **Proposed U-Net + Adaptive Polyphase Pooling** | **2–3 mins**    | **99.2%**| High       |
| FreeSurfer                            | 15 mins         | 95%      | Medium     |
| FSL                                   | 16 mins         | 93%      | Medium     |
| CAT                                   | 20 mins         | 90%      | Low        |

---

### 3. Visual Example

Below is a 2D slice from the side view of the original 3D volumetric MRI image (without skull-stripping), the ground-truth mask of the skull-stripped brain MRI, and the predicted mask from the proposed model using the Adaptive Polyphase Pooling Subsampling technique.

![image2](image2)

Fig. 6. displays a 2d slice of the side view of the original 3d volumetric MRI image without skull-stripping, the actual mask of the skull-stripped Brain MRI image and the predicted mask by the proposed model with the Adaptive Pooling Subsampling technique.

---

> **Conclusion:** Adaptive Polyphase Pooling eliminates shift sensitivity in CNNs, providing consistent, reliable skull-stripping across all MRI slice displacements.

---

## 🔗 Citation

```bibtex
@inproceedings{Padalia2023ShiftInvariantSkullStripping,
  author    = {A. Padalia and T. Shah and P. Pujari and A. Karande},
  title     = {Enhancing Shift-Invariance for Accurate Brain MRI Skull-Stripping using Adaptive Polyphase Pooling in Modified U-Net},
  booktitle = {2023 2nd International Conference on Automation, Computing and Renewable Systems (ICACRS)},
  year      = {2023},
  pages     = {1790--1798},
  doi       = {10.1109/ICACRS58579.2023.10404359},
  keywords  = {Image segmentation;Adaptive systems;Three-dimensional displays;Magnetic resonance imaging;Supervised learning;Feature extraction;Biomedical imaging;Adaptive Polyphase Pooling;U-net;Supervised-learning;skull-stripping;Brain Magnetic Resonance Imaging},
}
```

---

## 📄 License

This repository is intended for academic/research use. Please cite the above paper when using the code or methodology.

---
