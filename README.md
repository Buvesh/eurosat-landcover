# 🛰️ Land-Cover Classification on Sentinel-2 Imagery (EuroSAT)

## 🎯 Project Goal
Classify Sentinel-2 satellite observation patches across 10 distinct land-use/land-cover categories using PyTorch. This project compares the efficacy of transfer learning against scratch training and evaluates the performance boost of engineering 14-channel multispectral feature representations for Earth observation tasks.

## 📊 Dataset Overview
* **Source:** EuroSAT RGB & Multispectral (13 Bands)
* **Scale:** 27,000 Sentinel-2 patches ($64 \times 64$ pixels) (Helber et al., 2019).
* **10 Target Classes:** 🌾 Annual Crop | 🌲 Forest | 🌿 Herbaceous Vegetation | 🛣️ Highway | 🏭 Industrial | 🐄 Pasture | 🌳 Permanent Crop | 🏘️ Residential | 🏞️ River | 🌊 Sea Lake

## 🧠 Methodology & Architecture
* **Model Architecture:** ResNet18. For the multispectral experiment, the foundational `conv1` layer was dynamically modified to accept a 14-channel tensor input.
* **Validation Strategy:** Fixed 80/20 stratified split (SEED=42) to maintain strict class balance.
* **Optimization Parameters:** AdamW optimizer ($lr = 3 \times 10^{-4}$), Cross-Entropy Loss, batch size of 64.
* **Feature Engineering:** Dynamic, pixel-wise calculation of the Normalized Difference Vegetation Index (NDVI), concatenated as an explicit 14th input channel to isolate vegetation health:
$$NDVI = \frac{NIR - Red}{NIR + Red + 10^{-6}}$$

## 📊 Experimental Results
*Note: Every metric listed below represents a fully converged run completed on a Tesla T4 GPU.*

| Experiment | Input Format | Initialization | Test Accuracy | Epochs |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 ResNet18 (Baseline) | RGB (3 Channels) | ImageNet Pretrained | 95.81% | 5 |
| 🥈 ResNet18 (Multispectral) | 13 Bands + NDVI (14 Channels) | Random (Scratch) | 90.04% | 5 |
| 🥉 ResNet18 (Baseline) | RGB (3 Channels) | Random (Scratch) | 88.43% | 5 |

## 💡 Key Insights & Error Analysis
* **Transfer Learning Superiority:** Fine-tuning ImageNet weights achieved a dominant 95.81% test accuracy, severely outperforming the scratch RGB baseline (88.43%). This highlights the immense value of low-level visual priors (edges, gradients, textures) transferred from natural imagery.
* **Impact of Multispectral Features & NDVI:** Training from scratch on 14-channel multispectral data boosted accuracy to 90.04% (+1.61% over the RGB scratch model). Giving the model access to invisible spectrums (like Near-Infrared) significantly improved its ability to differentiate between visually similar vegetation classes, such as Herbaceous Vegetation and Pasture.
* **Primary Misclassifications:** A review of the off-diagonal errors reveals persistent, logical confusion between dense vegetation categories (Permanent Crop vs Annual Crop) and grey-surface infrastructure (Highway vs Industrial).

## 🚧 Limitations & Future Work
* **Spatial Leakage Risk:** Standard random patch-level splitting may introduce spatial autocorrelation (adjacent pixels from the same geographic tile bleeding into both train and test sets). Future iterations will implement Spatial Cross-Validation grouping.
* **Transition to Segmentation:** While global image classification works for isolated patches, operational land-cover mapping requires pixel-dense boundaries. Future developments will transition this architecture to a U-Net semantic segmentation pipeline.

## 💻 How to Run
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Buvesh/eurosat-landcover.git](https://github.com/Buvesh/eurosat-landcover.git)
cd eurosat-landcover
pip install torch torchvision scikit-learn matplotlib datasets
jupyter notebook
