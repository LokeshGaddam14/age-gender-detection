"""Age and Gender Detection Model

This module contains the main model architecture and inference code
for detecting age and gender from facial images.

Author: Data Science Team
Date: November 2025
"""

import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings('ignore')


class AgeGenderDetectionModel:
    """
    Multi-output neural network for simultaneous age and gender prediction.
    
    Attributes:
        model (keras.Model): The compiled neural network model
        input_shape (tuple): Input image shape (height, width, channels)
        gender_classes (int): Number of gender classes (default: 2)
    """
    
    def __init__(self, input_shape=(128, 128, 3), gender_classes=2):
        """Initialize the Age and Gender Detection Model.
        
        Args:
            input_shape (tuple): Shape of input images
            gender_classes (int): Number of gender classes
        """
        self.input_shape = input_shape
        self.gender_classes = gender_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the multi-output neural network architecture.
        
        Returns:
            keras.Model: Compiled model ready for training
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional blocks
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Flatten and Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Gender output (binary classification)
        gender_output = layers.Dense(self.gender_classes, 
                                    activation='softmax',
                                    name='gender')(x)
        
        # Age output (regression or classification)
        age_output = layers.Dense(1, activation='relu', name='age')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, 
                           outputs=[gender_output, age_output])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'gender': 'categorical_crossentropy', 'age': 'mse'},
            loss_weights={'gender': 1.0, 'age': 0.5},
            metrics={'gender': 'accuracy', 'age': 'mae'}
        )
        
        return model
    
    def train(self, X_train, y_gender_train, y_age_train, 
             X_val=None, y_gender_val=None, y_age_val=None,
             epochs=50, batch_size=32):
        """Train the model on provided data.
        
        Args:
            X_train (np.array): Training images
            y_gender_train (np.array): Training gender labels
            y_age_train (np.array): Training age values
            X_val (np.array): Validation images
            y_gender_val (np.array): Validation gender labels
            y_age_val (np.array): Validation age values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.callbacks.History: Training history
        """
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data
        validation_data = None
        if X_val is not None:
            validation_data = (X_val, [y_gender_val, y_age_val])
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, [y_gender_train, y_age_train], 
                        batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_data,
            verbose=1
        )
        
        return history
    
    def predict(self, image):
        """Predict age and gender for a single image.
        
        Args:
            image (np.array or str): Image array or path to image file
            
        Returns:
            tuple: (predicted_age, predicted_gender)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        gender_pred, age_pred = self.model.predict(image, verbose=0)
        
        # Get predicted gender
        gender_class = np.argmax(gender_pred[0])
        gender_labels = ['Male', 'Female']
        predicted_gender = gender_labels[gender_class]
        
        predicted_age = age_pred[0][0]
        
        return predicted_age, predicted_gender
    
    def save(self, filepath):
        """Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Age and Gender Detection Model initialized")
    print("Model architecture: Multi-output CNN")
    print("Gender classes: 2 (Male, Female)")
    print("Age output: Continuous regression")
