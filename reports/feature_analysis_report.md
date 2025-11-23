# Feature Analysis Report

## Dataset Overview

- **Total Samples**: 7442
- **Total Features**: 69
- **Metadata Columns**: 7

## Feature Quality

- **Missing Values**: 0 (0.00%)
- **Infinite Values**: 0
- **Low Variance Features**: 57 (variance < 0.01)

## Feature Statistics

### Summary Statistics
- **Mean**: Features range from -27.1399 to 3.0448
- **Std**: Features range from 0.0000 to 1.5356
- **Min**: Features range from -99.0018 to 1.2680
- **Max**: Features range from -23.4348 to 5.0050

## Class Imbalance Analysis


- **Imbalance Ratio**: 1.17 (max/min percentage)
- **Most Common Emotion**: Anger (17.1%)
- **Least Common Emotion**: Neutral (14.6%)

✓ Dataset is relatively balanced

## Feature Scaling

- **Scale Ratio**: 194350.09 (max range / min range)
- ⚠️ **Recommendation**: Apply scaling/normalization (StandardScaler or MinMaxScaler)

## Low Variance Features

- **Features with variance < 0.01**: 57

### Low Variance Features to Consider Removing:
- `zcr_mean`: variance = 0.000587
- `energy_mean`: variance = 0.000073
- `energy_entropy_mean`: variance = 0.002663
- `spectral_centroid_mean`: variance = 0.000464
- `spectral_spread_mean`: variance = 0.000061
- `spectral_flux_mean`: variance = 0.000001
- `spectral_rolloff_mean`: variance = 0.001692
- `mfcc_11_mean`: variance = 0.009420
- `mfcc_12_mean`: variance = 0.007486
- `mfcc_13_mean`: variance = 0.007095

## Feature Importance (Mutual Information)

### Top 15 Most Important Features:

1. `mfcc_2_mean`: 0.2071
1. `delta energy_mean`: 0.1827
1. `spectral_entropy_mean`: 0.1528
1. `mfcc_1_mean`: 0.1510
1. `chroma_3_mean`: 0.1258
1. `mfcc_3_mean`: 0.1103
1. `zcr_mean`: 0.1084
1. `spectral_rolloff_mean`: 0.1034
1. `mfcc_4_mean`: 0.0954
1. `energy_entropy_mean`: 0.0950
1. `chroma_std_mean`: 0.0940
1. `spectral_centroid_mean`: 0.0928
1. `chroma_8_mean`: 0.0727
1. `chroma_7_mean`: 0.0726
1. `chroma_9_mean`: 0.0533

## Correlation Analysis

- **Highly Correlated Pairs (|r| > 0.9)**: 4

### Top Highly Correlated Pairs
- `zcr_mean` <-> `spectral_centroid_mean`: 0.971
- `zcr_mean` <-> `spectral_rolloff_mean`: 0.977
- `spectral_centroid_mean` <-> `spectral_rolloff_mean`: 0.962
- `spectral_entropy_mean` <-> `spectral_rolloff_mean`: 0.917

## Feature Distribution by Emotion

Features show different distributions across emotion classes, which is expected for effective classification.

## PCA Analysis

Principal Component Analysis shows:
- **PC1 Explained Variance**: 10.3%
- **PC2 Explained Variance**: 6.9%
- **Total Explained Variance (2 components)**: 17.2%

## Recommendations

- ⚠️ **Low PCA Variance**: Features may need dimensionality reduction
- ⚠️ **Low Variance Features**: Remove features with variance < 0.01
- ⚠️ **Feature Scaling**: Apply StandardScaler or MinMaxScaler before training
- ✓ Features are ready for model training after addressing above recommendations
