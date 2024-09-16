# Machine Learning for Pneumonia Detection: Convolutional Neural Networks In Image Recognition





## Table of Contents
[Click To View Code<<<<<<<<<<<<<<<<<<<<<<<<<](#code)

0. [What Have I Learned ](#0-What-Have-I-Learned)
1. [Introduction](#1-introduction)

2. [Initial Version (Pre-Improvements)](#2-initial-version-pre-improvements)

3. [Improved Version (Post-Improvements)](#3-improved-version-post-improvements)

4. [Analysis of Changes](#4-analysis-of-changes)

	- [4.1 Data Augmentation](#41-data-augmentation)

	- [4.2 Fine-tuning](#42-fine-tuning)

	- [4.3 Learning Rate](#43-learning-rate)

	- [4.4 Epochs](#44-epochs)

5. [Conclusion](#5-conclusion)

[Data set: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data]

## What Have I Learned
1. **Deep Learning Frameworks & Model Architecture - Convolutional Neural Networks, TensorFlow/Keras,**
2. **Model Optimization - Fine-tuning, Data Augmentation, Hyperparameter Tuning,**
3. **Performance Metrics & Model Evaluation - Accuracy and Loss Analysis,**
4. **Training Process Management & Evaluation - Early Stopping, Performance Metrics Analysis**
5. **Validation Techniques - Cross-Validation**

## 1. Introduction
This report examines two iterations of custom code developed for a Pneumonia Recognition project utilizing the pre-trained VGG16 model. The project focuses on assessing the effects of fine-tuning, data augmentation, learning rate adjustments, and epoch changes on model performance. Using a dataset of chest X-rays, the report outlines the enhancements made from the initial to the refined code version and provides an analysis of their impact on performance.

## 2. Initial Version (Pre-Improvements)
In the initial code version, basic data augmentation was applied to the training set, and the VGG16 model was used without any fine-tuning. The model was trained for a limited number of epochs with a relatively high learning rate.

![Training_Pre_Adjustments.png](https://raw.githubusercontent.com/Xenizal/Python_Project/main/Training_Pre_Adjustments.png)


**Key Aspects:**
- **Model:** VGG16, with all layers frozen (no fine-tuning)
- **Data Augmentation:**
  - Rotation range: 20 degrees
  - Width/Height shift: 0.2
  - Shear/Zoom range: 0.2
  - Brightness adjustment: [0.8, 1.2]
  - Channel shift: 20.0
  - Vertical flip: Enabled
- **Learning Rate:** 0.001
- **Epochs:** 3
- **Callbacks:** Early stopping with patience = 2, restoring the best model weights.

**Code Snippet:**
```python
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=3, validation_data=val_generator, callbacks=[early_stopping, model_checkpoint])
```

**Results:**

-   **Test Accuracy:** ~80.6%
-   **Test Loss:** ~0.4658

![Results_Pre_Adjustments.png](https://raw.githubusercontent.com/Xenizal/Python_Project/main/Results_Pre_Adjustments.png)


## 3. Improved Version (Post-Improvements)

In the improved version, several changes were made to improve the model’s accuracy and generalization capabilities. These changes included fine-tuning, more aggressive data augmentation, lowering the learning rate, and increasing the number of epochs.

![Training_Post_Adjustments.png](https://raw.githubusercontent.com/Xenizal/Python_Project/main/Training_Post_Adjustments.png)

**Key Aspects:**

-   **Model:** VGG16, with the last 4 layers unfrozen (fine-tuning enabled)
-   **Data Augmentation:**
    -   More aggressive transformations for better generalization:
        -   Rotation range: Increased to 25 degrees
        -   Width/Height shift: Increased to 0.3
        -   Shear/Zoom range: Increased to 0.3
        -   Brightness adjustment: [0.7, 1.3]
        -   Channel shift: Increased to 30.0
        -   Vertical flip: Enabled
-   **Learning Rate:** Reduced to 0.0001 to allow more refined updates.
-   **Epochs:** Increased to 10 to give the model more time to learn.
-   **Callbacks:** Early stopping with patience = 3, restoring the best model weights.

**Code Snippet:**

```python
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping, model_checkpoint])` 
```

**Results:**

-   **Test Accuracy:** ~80.6%
-   **Test Loss:** ~0.4658

![Results_Post_Adjustments.png](https://raw.githubusercontent.com/Xenizal/Python_Project/main/Results_Post_Adjustments.png)

## 4. Analysis of Changes

### 4.1 Data Augmentation

The improved version includes more aggressive data augmentation. This technique was used to prevent overfitting by artificially expanding the training set through transformations such as larger rotations, width/height shifts, and brightness adjustments. However, these changes did not significantly affect the final test accuracy.

### 4.2 Fine-tuning

Fine-tuning the last four layers of the VGG16 model aimed to allow the model to learn more dataset-specific features while keeping the majority of the layers frozen to preserve the general features learned from the dataset. This strategy, combined with a reduced learning rate, helps the model make more precise updates. Despite these changes, there was no notable improvement in accuracy.

### 4.3 Learning Rate

Reducing the learning rate from 0.001 to 0.0001 allowed for finer updates to the model’s weights. However, since the test accuracy remained unchanged, it suggests that the learning rate adjustment alone was not enough to improve the model’s performance on this dataset.

### 4.4 Epochs

Increasing the number of epochs to 10 gave the model more opportunities to learn. However, early stopping was employed to prevent overfitting, and the model did not train for all 10 epochs. The results indicate that increasing epochs did not lead to a significant change in performance.

## 5. Conclusion

After implementing several changes aimed at improving the model’s accuracy, including data augmentation, fine-tuning, lowering the learning rate, and increasing the number of epochs, the performance gains were negligible. The final test accuracy of the model remained at approximately 80.6%, with a loss of 0.4658, both before and after the improvements.

While the changes did not produce a significant improvement in this case, the project demonstrates the process of model experimentation and optimization. Fine-tuning, adjusting the learning rate, and experimenting with data augmentation.

This project served a valuable learning experience for me, showcasing the importance of iterative model improvement and validation.

-    Experimenting with **other** pre-trained models such as ResNet50 or InceptionV3 could yield better results,
-    Further exploration of **hyperparameters** such as batch size and optimizer settings might lead to performance improvements,
-    Using an **ensemble of models** may help achieve better generalization and higher accuracy.

## CODE:
```python
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths to the directories
train_dir = r"Python_Project\Pneumonia_Recognition\chest_xray\train"
val_dir = r"Python_Project\Pneumonia_Recognition\chest_xray\val"
test_dir = r"Python_Project\Pneumonia_Recognition\chest_xray\test"

# Data Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,  
    width_shift_range=0.3,  
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.7, 1.3],  
    channel_shift_range=30.0, 
    fill_mode="nearest",
)

# Data preprocessing for validation and test data
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Creating data generators for training, validation and testing
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=256, class_mode="binary"
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=256, class_mode="binary"
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=256, class_mode="binary"
)

# Model definition
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

# Fine-tuning: Unfreezing the last few layers of the base model
for layer in base_model.layers[:-4]:  # Unfreeze the last X layers you choose
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Learning rate for fine-tuning adjustment
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Early stopping and model saving
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)  
model_checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"
)

# Model training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint],
)

# Saving the training history to a file
with open("history_all.pkl", "wb") as f:
    pickle.dump([history.history], f)

# Saving the trained model
model.save("final_model_fine_tuned.h5")

```
