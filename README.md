# Age and Gender Detection ML Model

## Project Overview
This project implements a machine learning model that detects and predicts the age and gender of individuals from images. It uses convolutional neural networks (CNN) to process image data and make accurate predictions.

## Objective
- Build a multi-output neural network that can simultaneously predict age and gender
- Achieve high accuracy in both age and gender classification
- Create a deployable model that can work with real-world image inputs
- Enable real-time predictions on webcam streams

## Technologies Used
- **Python 3.x**
- **TensorFlow / Keras** - Deep Learning Framework
- **OpenCV** - Image Processing
- **NumPy & Pandas** - Data Manipulation
- **Matplotlib & Seaborn** - Data Visualization
- **Scikit-learn** - ML utilities

## Dataset
- Image dataset with labeled age and gender information
- Preprocessed and normalized for model training
- Data split: Training (80%), Validation (10%), Testing (10%)
- Age ranges: Continuous or categorical predictions
- Gender: Binary classification (Male/Female)

## Model Architecture
- **Input**: RGB images (224x224 or 128x128 pixels)
- **Backbone**: Pre-trained CNN architecture (VGG16, ResNet50, or MobileNet)
- **Feature Extraction**: Multiple convolutional layers with pooling
- **Output Layer 1**: Gender (Binary Classification)
- **Output Layer 2**: Age (Regression or Classification)
- **Loss Function**: Binary Crossentropy (Gender) + MSE/Categorical Crossentropy (Age)

## Project Structure
```
age-gender-detection/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   └── age_gender_model.h5
├── notebooks/
│   └── age_gender_detection.ipynb
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   └── utils.py
├── results/
│   ├── predictions.csv
│   └── visualizations/
├── requirements.txt
└── README.md
```

## How to Use

### Installation
```bash
pip install -r requirements.txt
```

### Training
```python
from src.model import AgeGenderModel

model = AgeGenderModel()
model.train(train_data_path, validation_data_path)
model.save('models/age_gender_model.h5')
```

### Inference
```python
import cv2
from src.model import AgeGenderModel

model = AgeGenderModel()
model.load('models/age_gender_model.h5')

image = cv2.imread('test_image.jpg')
age, gender = model.predict(image)
print(f"Age: {age}, Gender: {gender}")
```

### Real-time Webcam
```python
from src.model import AgeGenderModel
import cv2

model = AgeGenderModel()
model.load('models/age_gender_model.h5')
model.predict_webcam()
```

## Results
- **Gender Classification Accuracy**: [To be filled]
- **Age Prediction MAE**: [To be filled]
- **Model Size**: [To be filled]
- **Inference Time**: [To be filled]

## Key Features
- ✅ Multi-output neural network
- ✅ Real-time predictions
- ✅ Pre-trained model weights
- ✅ Data augmentation
- ✅ Transfer learning support

## Future Enhancements
- Increase dataset diversity for better generalization
- Implement real-time predictions using webcam
- Deploy as a web application
- Mobile app development
- Improve age prediction for edge cases
- Add emotion detection

## Performance Metrics
- **Precision**: [To be calculated]
- **Recall**: [To be calculated]
- **F1-Score**: [To be calculated]
- **Validation Accuracy**: [To be calculated]

## Requirements
See `requirements.txt` for all dependencies.

## License
This project is open source and available for educational purposes.

## Contact
For questions or collaboration opportunities, please contact: [Your Email]

---
**Status**: Active Development  
**Last Updated**: November 2025
