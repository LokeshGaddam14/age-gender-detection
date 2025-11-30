# Data Directory

## Overview
This directory contains data files used for the Age-Gender Detection model.

## Dataset Information

### Data Source
- **Source**: UTKFace dataset (publicly available)
- **Description**: Large-scale face dataset with age, gender, and ethnicity labels
- **Size**: Over 20,000 face images
- **Format**: JPEG images

### Data Structure
```
data/
├── raw/
│   └── age_gender_images/  # Original dataset images
├── processed/
│   └── train_test_split/   # Preprocessed and split data
└── metadata/
    └── image_labels.csv    # Image filenames with age, gender labels
```

## Labels
- **Age**: Continuous variable (0-100 years)
- **Gender**: Binary (0 = Male, 1 = Female)

## Data Preprocessing
Images are resized to 128x128 pixels and normalized to [0, 1] range.

## Usage
The `AgeGenderDetectionModel` class loads and preprocesses images from this directory.

## Licensing
Please refer to the original UTKFace dataset repository for licensing information.
